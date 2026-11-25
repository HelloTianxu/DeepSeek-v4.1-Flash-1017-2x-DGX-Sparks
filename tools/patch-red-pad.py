#!/usr/bin/env python3
"""E-BANK-01 实验A：消除 down 内核 shared store 的 8 路 bank 冲突（padding，位一致）。

机制（源码实读 + 算术推导，非猜测）：
  experts_grouped.cuh:433  const int g8 = lane >> 2, t = lane & 3;
  experts_grouped.cuh:512  const int col = i * 16 + h * 8 + 2 * t;
  experts_grouped.cuh:513  red[warp][g8][col] = ...
  声明 :462  __shared__ float red[W][16][NT * 16];
  生产 NT=8 ⇒ row stride = 128 float = 512 B = 32 banks × 4 B 的整数倍
  ⇒ bank = ((warp*16+g8)*128 + col) % 32 = col % 32，与 g8 无关
  ⇒ 一个 warp 里 lane%4 相同的 8 个线程（g8=0..7）落同一 bank ⇒ 8 路冲突
  实测（E-SELF-07 pipe-rows.csv）：bank_conflicts_st 32,982,485 / shared_st 指令 5,482,600 = 6.02×
  （低于 8 是因 live 过滤与部分 tile 空转）

修法：row stride 128 → 129（PAD=1）。129 % 32 = 1 ⇒ g8=0..7 分散到 8 个不同 bank。
**只改布局，不改任何数值、不改求和顺序（warp 顺序累加不变）⇒ 应位一致。**
读侧 `row = idx/(NT*16), col = idx%(NT*16)` 是逻辑索引，C 数组自动用物理 stride ⇒ 无需改。
shared 增量：W*16*1 float = 4*16*4 = 256 B（33920 → 34176，屋顶 101376，且 down REG 255 已钉 blocks=1）⇒ 占用率不变。

开关走 GroupedArgs 运行时字段 + 每次 launch 读 getenv（**不用 static**，否则同进程无法 A/B）。
只给生产分支（nt=8,warps=4,pf=1,g=2）加 PAD 实例，共 6 个（3 ranges × 2 fold），编译增量小。

用法：python3 patch-red-pad.py <tree_root>   # 幂等
"""
import pathlib, sys, os
PAD_N = os.environ.get("TF_RED_PAD_N", "2")   # 2 ⇒ stride+2 保 8B 对齐(STS.64)且 bank 全错开

root = pathlib.Path(sys.argv[1])
cuh = root / "experts_grouped.cuh"
cu = root / "experts.cu"
t = cuh.read_text()
t = "#define TF_RED_PAD_N " + PAD_N + "\n" + t

if "TF_EXL3_RED_PAD_APPLIED" in t:
    print("PATCH_ALREADY_APPLIED")
    sys.exit(0)

# ---- 1) 内核模板加 PAD 参数（默认 0 ⇒ 既有实例化语义不变）----
A_OLD = ("template <int CB, int NT, int W, int PF, int LO, int HI, int G, bool FOLD>\n"
         "__global__ void __launch_bounds__(W * 32) grouped_rows_kernel(")
A_NEW = ("// TF_EXL3_RED_PAD_APPLIED: PAD pads red's row stride to break the 8-way shared-store bank conflict.\n"
         "template <int CB, int NT, int W, int PF, int LO, int HI, int G, bool FOLD, int PAD = 0>\n"
         "__global__ void __launch_bounds__(W * 32) grouped_rows_kernel(")
assert t.count(A_OLD) == 1, f"锚点1 计数 {t.count(A_OLD)}"
t = t.replace(A_OLD, A_NEW)

# ---- 2) red 声明加 PAD（只改 prompt 的 :462，不动 decode 的 :380）----
B_OLD = ("        for (int j = 0; j < TPT; ++j) tot[g][j] = 0.f;\n"
         "    __shared__ float red[W][16][NT * 16];")
B_NEW = ("        for (int j = 0; j < TPT; ++j) tot[g][j] = 0.f;\n"
         "    __shared__ float red[W][16][NT * 16 + PAD];")
assert t.count(B_OLD) == 1, f"锚点2 计数 {t.count(B_OLD)}"
t = t.replace(B_OLD, B_NEW)

# ---- 3) GroupedArgs 加运行时开关 ----
C_OLD = "    int nwork = 0;\n};"
C_NEW = "    int nwork = 0;\n    int red_pad = 0;   // E-BANK-01: 1 ⇒ red row stride +1 (bank-conflict padding)\n};"
assert t.count(C_OLD) == 1, f"锚点3 计数 {t.count(C_OLD)}"
t = t.replace(C_OLD, C_NEW)

# ---- 4) dispatch：加 PAD 宏族 + 生产分支按 a.red_pad 二选一 ----
D_OLD = """    else if (a.nt == 8 && a.warps == 4 && a.pf == 1 && a.g == 2) { TF_RANGES(8, 4, 1, 2) }"""
D_NEW = """    else if (a.nt == 8 && a.warps == 4 && a.pf == 1 && a.g == 2) {
        if (a.red_pad) { TF_RANGES_P(8, 4, 1, 2) } else { TF_RANGES(8, 4, 1, 2) }
    }"""
assert t.count(D_OLD) == 1, f"锚点4 计数 {t.count(D_OLD)}"
t = t.replace(D_OLD, D_NEW)

# 宏族必须插到**使用点之前**（C 预处理器顺序展开）：锚在第一个 dispatch 分支之前
E_OLD = "    if (a.nt == 8 && a.warps == 4 && a.pf == 1 && a.g == 1) { TF_RANGES(8, 4, 1, 1) }"
E_NEW = """#define TF_LAUNCH_F_P(NT_, W_, PF_, G_, LO_, HI_, F_)                                                         \\
    grouped_rows_kernel<CB, NT_, W_, PF_, LO_, HI_, G_, F_, TF_RED_PAD_N><<<grid, W_ * 32, 0, stream>>>(                  \\
        a.x0, a.x1, a.tp0, a.tp1, a.k2_0, a.k2_1, a.uids, a.ucount, a.members, a.z, a.K, a.N, a.P, a.SK, a.maxm, \\
        a.slots, nz, a.work)
#define TF_LAUNCH_P(NT_, W_, PF_, G_, LO_, HI_)                                                               \\
    do {                                                                                                      \\
        if (a.fold) TF_LAUNCH_F_P(NT_, W_, PF_, G_, LO_, HI_, true);                                          \\
        else TF_LAUNCH_F_P(NT_, W_, PF_, G_, LO_, HI_, false);                                                \\
    } while (0)
#define TF_RANGES_P(NT_, W_, PF_, G_)                                                                         \\
    if (a.lo == 8 && a.hi == 8) TF_LAUNCH_P(NT_, W_, PF_, G_, 8, 8);                                          \\
    else if (a.lo >= 2 && a.hi <= 10) TF_LAUNCH_P(NT_, W_, PF_, G_, 2, 10);                                   \\
    else TF_LAUNCH_P(NT_, W_, PF_, G_, 2, 16);
    if (a.nt == 8 && a.warps == 4 && a.pf == 1 && a.g == 1) { TF_RANGES(8, 4, 1, 1) }"""
assert t.count(E_OLD) == 1, f"锚点5 计数 {t.count(E_OLD)}"
t = t.replace(E_OLD, E_NEW)

# 末尾补 #undef（原 #undef 块保持不动）
G_OLD = "#undef TF_RANGES\n#undef TF_LAUNCH\n#undef TF_LAUNCH_F\n}"
G_NEW = "#undef TF_RANGES_P\n#undef TF_LAUNCH_P\n#undef TF_LAUNCH_F_P\n#undef TF_RANGES\n#undef TF_LAUNCH\n#undef TF_LAUNCH_F\n}"
assert t.count(G_OLD) == 1, f"锚点7 计数 {t.count(G_OLD)}"
t = t.replace(G_OLD, G_NEW)
cuh.write_text(t)

# ---- 5) host 侧每次 launch 读 env（非 static，保证同进程可切换）----
u = cu.read_text()
F_OLD = ("    a.nwork = (int)(work.numel() / 2);\n"
         "    auto stream = at::cuda::getCurrentCUDAStream();\n"
         "    if (cb == 0) tf_exl3x::grouped_rows_launch<0>(a, stream);")
F_NEW = ("    a.nwork = (int)(work.numel() / 2);\n"
         "    { const char* v = getenv(\"TF_EXL3_RED_PAD\"); a.red_pad = (v && v[0] == '1') ? 1 : 0; }\n"
         "    auto stream = at::cuda::getCurrentCUDAStream();\n"
         "    if (cb == 0) tf_exl3x::grouped_rows_launch<0>(a, stream);")
assert u.count(F_OLD) == 1, f"锚点6 计数 {u.count(F_OLD)}"
u = u.replace(F_OLD, F_NEW)
cu.write_text(u)
print("PATCH_OK")