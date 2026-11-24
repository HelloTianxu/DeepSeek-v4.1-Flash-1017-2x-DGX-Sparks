#!/usr/bin/env python3
"""稠密线性链路 + 集合通信的物理账：从 trace 实测时长 + config 真实维度算达成率与上界。

形状判定不是猜的，用两个独立分解互相校验：
  _gemm grid = ceil(M/128) * (N/128)          （tri_markov.zig:163, gemm_bn=128, bm=128）
  rot_in<void> grid = (ceil(K/512), M, 1)
  按 K 分组的 rot_in 调用数必须等于同 K 的 _gemm 调用数 —— 实测 400/185/100/10 逐一相等。
维度来源：zig/src/families/dsv41/config.zig 的 test_config（我方 checkpoint）。
硬件屋顶：FP16 GEMM 100.82 TFLOPS（实测）、DRAM 246.2 GB/s（实测）、
          RoCE 2×200 Gb/s ACTIVE = 50 GB/s/方向（ibv_devinfo + sysfs rate 实读）。
"""
import json, collections

TRACE = "runs/forge-20261123-steady-trace/timeline.json"
K = json.load(open(TRACE))["kernels"][6104:12184]
W = max(x[1] for x in K) - min(x[0] for x in K)

FP16_ROOF = 100.82e12
DRAM_ROOF = 246.2e9
LINK = 50e9          # 2 x 200 Gb/s, 单方向聚合
HIDDEN = 5120

# grid -> (M, K, N, 说明)；M 由 grid/(N/128) 反解，K 由 rot_in 分块计数交叉校验
SHAPE = {
    2048: (2048, 1280, 16384, "q_b_proj  (TP2 本地 64x512/2)"),
    640:  (2048, 4096, 5120,  "o_proj    (64x512/8 -> d)"),
    128:  (2048, 4096, 1024,  "o_lora down (o_lora_rank)"),
    3200: (2048, 3072, 25600, "宽投影 K=3072"),
    160:  (2048, 5120, 1280,  "q_a_proj  (d -> q_lora_rank)"),
    64:   (2048, 5120, 512,   "kv/rope 支路"),
    512:  (436,  1280, 16384, "q_b_proj  尾块 M=436"),
    800:  (436,  3072, 25600, "宽投影   尾块 M=436"),
    40:   (436,  5120, 1280,  "q_a_proj  尾块 M=436"),
}

g = collections.defaultdict(lambda: [0, 0])
for s, e, st, n, gx, gy, gz in K:
    if n == "_gemm":
        g[gx][0] += e - s; g[gx][1] += 1

print("== _gemm 逐形状达成率（分母 FP16 实测屋顶 100.82 TFLOPS）==")
print(f"{'grid':>6}{'次':>5}{'M':>6}{'K':>6}{'N':>7}{'总ms':>9}{'主TFLOPS':>10}{'含Had':>9}{'fp16权重GB':>11}  说明")
tf = tt = tb = 0.0
for gx, (d, c) in sorted(g.items(), key=lambda kv: -kv[1][0]):
    if gx not in SHAPE:
        print(f"{gx:>6}{c:>5}{'?':>6}{'?':>6}{'?':>7}{d/1e6:>9.2f}{'—':>10}{'—':>9}{'—':>11}  小计未定形（占窗口 {100*d/W:.2f}%）")
        tt += d; continue
    M, Kk, N, note = SHAPE[gx]
    fl = 2 * M * Kk * N * c
    had = 2 * M * 128 * N * c            # out = (xh wq) H SCALE svh，每 128 列块一次 128x128
    by = 2 * Kk * N * c
    print(f"{gx:>6}{c:>5}{M:>6}{Kk:>6}{N:>7}{d/1e6:>9.2f}{fl/d*1e9/1e12:>10.2f}{(fl+had)/d*1e9/1e12:>9.2f}{by/1e9:>11.3f}  {note}")
    tf += fl + had; tt += d; tb += by
print(f"合计 {tt/1e6:.2f} ms（{100*tt/W:.2f}% 窗口）  加权达成 {tf/tt*1e9/1e12:.2f} TFLOPS = {100*tf/tt*1e9/FP16_ROOF:.1f}% 屋顶")
print(f"unpack 物化的 fp16 权重 {tb/1e9:.3f} GB/请求 ⇒ 写+读往返 {2*tb/1e9:.3f} GB，按 DRAM 屋顶至少 {2*tb/DRAM_ROOF*1e3:.1f} ms")
print(f"同源 EXL3 2.9bpw 只需 {tb*2.9/16/1e9:.3f} GB ⇒ 直读 trellis 可省 {2*tb/1e9 - tb*2.9/16/1e9:.2f} GB 流量（{2*16/2.9:.1f}x）")

print()
print("== 稠密线性链路总账（三行必须合看）==")
tot = 0
for label, pat in [("_gemm", lambda n: n == "_gemm"),
                   ("unpack (全部)", lambda n: "unpack_kernel" in n),
                   ("rot_in<void>", lambda n: "rot_in_kernel(const void" in n)]:
    d = sum(e - s for s, e, st, n, *_ in K if pat(n))
    tot += d
    print(f"  {label:<16}{d/1e6:>9.2f} ms{100*d/W:>8.2f}% 窗口")
print(f"  {'链路合计':<16}{tot/1e6:>9.2f} ms{100*tot/W:>8.2f}% 窗口")

print()
print("== AllGather（Comm.gather，[world,n,D] fp32）==")
ag = collections.defaultdict(lambda: [0, 0])
for s, e, st, n, gx, gy, gz in K:
    if "AllGather" in n: ag[gx][0] += e - s; ag[gx][1] += 1
for gx, (d, c) in sorted(ag.items(), key=lambda kv: -kv[1][0]):
    print(f"  grid {gx:>3}  {c:>4}次  总{d/1e6:>8.2f}ms  均{d/c/1e6:>7.4f}ms  {100*d/W:>5.2f}% 窗口")
d, c = ag[64]
msg = 2048 * HIDDEN * 4
rate = msg / (d / c / 1e9)
print(f"  主项消息 n*hidden*4 = 2048*{HIDDEN}*4 = {msg/1e6:.2f} MB/rank")
print(f"  达成 {rate/1e9:.2f} GB/s = 单口 200Gb/s 的 {100*rate/25e9:.1f}%，双口聚合 50GB/s 的 {100*rate/LINK:.1f}%")
print(f"  协议 RING_LL：8B 线元只载 4B 数据 ⇒ 有效上限约 {LINK/2/1e9:.1f} GB/s，当前占其 {100*rate/(LINK/2):.1f}%")
print(f"  若达单口线速 25GB/s：{msg/25e9*1e3:.3f} ms/次 ⇒ 主项 {msg/25e9*c*1e3:.1f} ms（现 {d/1e6:.1f}）")

print()
print("== 全局 Amdahl（底 2084.42 tok/s；份额取窗口口径）==")
BASE = 2084.42
def amd(share, factor, label):
    s = share / 100
    sp = 1 / (1 - s * (1 - factor))
    print(f"  {label:<40} 份额{share:>6.2f}%  该部件×{factor:<5.2f} ⇒ 全局×{sp:.4f}  {BASE*sp:>7.1f} tok/s (+{BASE*(sp-1):>5.1f})")
amd(100 * ag[64][0] / W, 0.50, "AllGather 主项减半（协议或字节）")
amd(100 * ag[64][0] / W, 0.00, "AllGather 主项归零（理论上界）")
amd(100 * tot / W, 0.70, "稠密链路 -30%（GEMM 达 90% 屋顶）")
amd(100 * tot / W, 0.00, "稠密链路归零（理论上界）")
amd(100 * 275e6 / W, 0.00, "每请求固定成本 275ms 归零")
amd(100 * (1282.39 + 626.15) * 1e6 / W, 0.90, "专家核心 -10%（E-SELF-10 已证极难）")