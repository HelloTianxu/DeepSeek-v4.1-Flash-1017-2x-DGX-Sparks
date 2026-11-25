#!/usr/bin/env python3
"""E-ROWS-01：把 grouped_mma 的 warp/行布局参数化，使 MMA_ROWS 可取 32。

## 为什么之前 R32 会 illegal memory access
原源码把「64 行 = 4 row-group × 16 行」和「256 线程 = 64 行 × 4 线程」写死在四处：
  1. mma_slice  crow = threadIdx.x >> 2      ⇒ crow ∈ 0..63，R32 时 rows_sh[32..63] 越界
  2. mma_slice  lrow = rg*16 + ...，rg=warp&3 ⇒ lrow 可达 63，As[..][63] 越界
  3. kernel     r0/r1 = rows_sh[rg*16+g8(+8)] ⇒ 同上
  4. cp.async   每行 4 线程 × CPR=KB/2 chunk  ⇒ R32 时每行 8 线程，chunk 索引与 src 偏移全错
（work 编码是第五处，但 m2_work(...,rows) 本就带 rows 参数，probe 传 32 即可，无需改源码。）

## 本 patch 的做法
引入派生常量，全部由 MMA_ROWS 推出，**R=64 时逐字等价于原式**：
  MMA_RG  = MMA_ROWS/16          row-group 数（原 4）
  MMA_CG  = 8/MMA_RG             col-group 数（原 2）
  MMA_NJ  = (MMA_COLS/16)/MMA_CG 每 warp 的 n16 tile 数（原 2）
  MMA_TPR = 256/MMA_ROWS         每 staged row 的线程数（原 4）
  CPR     = KB*16/(8*MMA_TPR)    每线程每步 chunk 数（原 KB/2）
warp 布局 rg=warp%MMA_RG, cg=warp/MMA_RG；累加器 tot/acc/sacc 由 [4][4] 改 [MMA_NJ*2][4]。

R=32 ⇒ RG=2, CG=4, NJ=1, TPR=8, CPR=1：8 warp = 2 row-group × 4 col-group，
每 warp 16 行 × 16 列，block 输出 32×64（原 64×64 的一半）⇒ MB 翻倍、总工作量不变。
shared: As[3][32][72]=13824 + Bs 16384 + rows_sh 128 = 30336 ⇒ 3 blocks（需 REG≤80，用 MINB=3）。

## 用法
  python3 patch-mma-rows.py <src_dir>          # 只参数化，R 默认 64（回归臂）
编译期用 -DTF_MMA_ROWS=32 -DTF_MMA_MINB=3 切换；同一模块可编两个 R 值做对照。
幂等：MARK 已存在则跳过。
"""
import sys, pathlib, re

MARK = "TF_MMA_ROWS_APPLIED"
src = pathlib.Path(sys.argv[1])
f = src / "experts_grouped.cuh"
t = f.read_text()
if MARK in t:
    print(f"SKIP already applied: {f}")
    raise SystemExit(0)

subs = []


def sub(old, new, why):
    subs.append((old, new, why))


# --- 1. MMA_ROWS 可配置 + MINB 宏 ---
sub("constexpr int MMA_ROWS = 64;        // member rows a program",
    """#ifndef TF_MMA_ROWS
#define TF_MMA_ROWS 64
#endif
#ifndef TF_MMA_MINB
#define TF_MMA_MINB 1
#endif
constexpr int MMA_ROWS = TF_MMA_ROWS;   // member rows a program""" + f"  // {MARK}",
    "MMA_ROWS 参数化")

# --- 2. 派生常量（放在 MMA_DEC 之后，此时 MMA_COLS 已定义）---
sub("constexpr int MMA_DEC = MMA_KB / 2;         // weight tiles a warp decodes a step (8 warps, 4 n16 tiles a k tile)",
    """constexpr int MMA_DEC = MMA_KB / 2;         // weight tiles a warp decodes a step (8 warps, 4 n16 tiles a k tile)
// warp/行布局全部由 MMA_ROWS 推出；MMA_ROWS=64 时 RG=4 CG=2 NJ=2 TPR=4，与原写死值逐字等价
constexpr int MMA_RG = MMA_ROWS / 16;             // row groups (warps along rows)
constexpr int MMA_CG = 8 / MMA_RG;                // col groups
constexpr int MMA_NJ = (MMA_COLS / 16) / MMA_CG;  // n16 tiles a warp
constexpr int MMA_TPR = 256 / MMA_ROWS;           // threads a staged row
static_assert(MMA_ROWS % 16 == 0 && MMA_RG >= 1 && 8 % MMA_RG == 0, "MMA_ROWS must divide 8 warps into 16-row groups");
static_assert((MMA_COLS / 16) % MMA_CG == 0, "n16 tiles must split across col groups");
static_assert(MMA_TPR >= 1 && (MMA_KB * 16) % (8 * MMA_TPR) == 0, "cp.async chunks must tile a row");""",
    "派生常量")

# --- 3. mma_slice 签名 tot 维度 ---
sub("uint4 (&bsh)[2][KB][4][32], float (&tot)[4][4]) {",
    "uint4 (&bsh)[2][KB][4][32], float (&tot)[MMA_NJ * 2][4]) {",
    "mma_slice 签名")

# --- 4. CPR ---
sub("constexpr int CPR = KB / 2;                           // chunks a thread a step",
    "constexpr int CPR = (KB * 16) / (8 * MMA_TPR);        // chunks a thread a step",
    "CPR")

# --- 5. crow/cpart ---
sub("const int crow = threadIdx.x >> 2, cpart = threadIdx.x & 3;",
    "const int crow = threadIdx.x / MMA_TPR, cpart = threadIdx.x % MMA_TPR;",
    "crow/cpart")

# --- 6. rg/cg（mma_slice 内）---
sub("const int rg = warp & 3, cg = warp >> 2;                  // its rows 16 rg .., its n16 tiles 2 cg, 2 cg + 1",
    "const int rg = warp % MMA_RG, cg = warp / MMA_RG;         // its rows 16 rg .., its n16 tiles MMA_NJ cg ..",
    "mma_slice rg/cg")

# --- 7. 预填充 cp.async ---
sub("cp_async16(&As[st][crow][(cpart + 4 * c) * 8], src + st * KB * 16 + 32 * c, cr >= 0);",
    "cp_async16(&As[st][crow][(cpart + MMA_TPR * c) * 8], src + st * KB * 16 + MMA_TPR * 8 * c, cr >= 0);",
    "cp.async 预填充")

# --- 8. 主循环 cp.async ---
sub("""cp_async16(&As[ahead % MMA_STAGES][crow][(cpart + 4 * c) * 8],
                               src + (size_t)ahead * KB * 16 + 32 * c, cr >= 0);""",
    """cp_async16(&As[ahead % MMA_STAGES][crow][(cpart + MMA_TPR * c) * 8],
                               src + (size_t)ahead * KB * 16 + MMA_TPR * 8 * c, cr >= 0);""",
    "cp.async 主循环")

# --- 9. acc/sacc 声明与初始化 ---
sub("""    float acc[4][4], sacc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[i][c] = sacc[i][c] = 0.f;""",
    """    float acc[MMA_NJ * 2][4], sacc[MMA_NJ * 2][4];
#pragma unroll
    for (int i = 0; i < MMA_NJ * 2; ++i)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[i][c] = sacc[i][c] = 0.f;""",
    "acc/sacc")

# --- 10. mma 内层 j 循环 + bsh 索引 ---
sub("""                for (int j = 0; j < 2; ++j) {
                    const uint4 b = bsh[buf][kk][2 * cg + j][lane];""",
    """                for (int j = 0; j < MMA_NJ; ++j) {
                    const uint4 b = bsh[buf][kk][MMA_NJ * cg + j][lane];""",
    "mma j 循环")

# --- 11. sacc 累加 ---
sub("""            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    sacc[i][c] = wk == 0 ? acc[i][c] : sacc[i][c] + acc[i][c];""",
    """            for (int i = 0; i < MMA_NJ * 2; ++i)
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    sacc[i][c] = wk == 0 ? acc[i][c] : sacc[i][c] + acc[i][c];""",
    "sacc 累加")

# --- 12. tot 写回 ---
sub("""                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int c = 0; c < 4; ++c) tot[i][c] = FOLD ? tot[i][c] + sacc[i][c] : sacc[i][c];""",
    """                for (int i = 0; i < MMA_NJ * 2; ++i)
#pragma unroll
                    for (int c = 0; c < 4; ++c) tot[i][c] = FOLD ? tot[i][c] + sacc[i][c] : sacc[i][c];""",
    "tot 写回")

# --- 13. launch_bounds ---
sub("__global__ void __launch_bounds__(256) grouped_mma_kernel(",
    "__global__ void __launch_bounds__(256, TF_MMA_MINB) grouped_mma_kernel(",
    "launch_bounds MINB")

# --- 14. kernel tot 声明与初始化 ---
sub("""    float tot[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int c = 0; c < 4; ++c) tot[i][c] = 0.f;""",
    """    float tot[MMA_NJ * 2][4];
#pragma unroll
    for (int i = 0; i < MMA_NJ * 2; ++i)
#pragma unroll
        for (int c = 0; c < 4; ++c) tot[i][c] = 0.f;""",
    "kernel tot")

# --- 15. kernel rg/cg ---
sub("    const int rg = warp & 3, cg = warp >> 2;\n",
    "    const int rg = warp % MMA_RG, cg = warp / MMA_RG;\n",
    "kernel rg/cg")

# --- 16. epilogue ---
sub("""    for (int i = 0; i < 4; ++i) {
        const int col = nt0 * 16 + cg * 32 + i * 8 + 2 * t;""",
    """    for (int i = 0; i < MMA_NJ * 2; ++i) {
        const int col = nt0 * 16 + cg * (MMA_NJ * 16) + i * 8 + 2 * t;""",
    "epilogue")

for old, new, why in subs:
    n = t.count(old)
    if n != 1:
        print(f"FAIL [{why}] 命中 {n} 次（须 1）:\n{old[:160]}")
        raise SystemExit(1)
    t = t.replace(old, new)

f.write_text(t)
print(f"APPLIED {len(subs)} 处 -> {f}")
for _, _, why in subs:
    print(f"  ok {why}")