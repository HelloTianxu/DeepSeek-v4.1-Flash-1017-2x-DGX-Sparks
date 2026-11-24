#!/usr/bin/env python3
"""E-SELF-08c：down 行 v1(grouped_rows) vs down3(cp.async 权重暂存) 同场 A/B。

这是木桶分析的第二个独立检验点：
  · 形状**就是生产形状** K=I=1152、N=D=5120、P=14336（down3 要求 N%256==0，5120/256=20 ✓）
  · down 的 warp 延迟首因是 long_scoreboard 35.9%（等内存），与 gate/up 的 barrier 首因不同
  · 若 down3 也把达成带宽顶上去 ⇒ 「片内递货是卡口」在两种不同 stall 画像上都成立

注意 down3 **不接受 work list**：grid=(ceil(maxm/64)=32, N/256=20, nexp_max=385)=246400 个程序，
靠 rows_sh[0]<0 早退；v1 走 820 项工作列表 ⇒ 40×820=32800。程序数差 7.5 倍，是 down3 的固有劣势。
down3 输出 bf16 Y[P,N] 且需 combine_y 而非 down_combine ⇒ **非 drop-in**，本轮只测内核本身。
"""
import os, json, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=8)
ap.add_argument("--arms", default="v1,down3,v1,down3")
ap.add_argument("--only", default="")
ap.add_argument("--ncu-warm", action="store_true")
ap.add_argument("--out", default="")
args = ap.parse_args()

import torch
from torch.utils.cpp_extension import load

BUILD = os.path.expanduser("~/ptxwork/probe-build")
SRC = os.path.expanduser("~/ptxwork/exl3src")
m = load(name="tf_exl3_probe",
         sources=[os.path.join(SRC, n) for n in
                  ["experts.cpp", "experts.cu", "experts_cb0.cu", "experts_cb1.cu", "experts_cb2.cu"]],
         extra_cflags=["-O3"],
         extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121,code=sm_121"],
         build_directory=BUILD, verbose=False)

dev = "cuda"
R, SLOTS = 2048, 7
P = R * SLOTS
K, N = 1152, 5120          # down：K=I、N=D，生产形状
E, MAXM = 385, 2048
MAXU = min(P, E)
K2 = 6
TW = 4 * K2
ND = (P + MAXU * 31) // 32
assert N % 256 == 0 and K % 64 == 0

WORDS = (K // 16) * (N // 16) * TW
STRIDE_B = WORDS * 4
buf = torch.randint(-2**31, 2**31 - 1, (E * WORDS,), dtype=torch.int32, device=dev)
ptrs = (buf.data_ptr() + torch.arange(E, dtype=torch.int64, device=dev) * STRIDE_B).contiguous()
k2 = torch.full((E,), K2, dtype=torch.int32, device=dev)

xd = torch.randn(P, K, dtype=torch.float16, device=dev)
Z = torch.zeros(P * N, dtype=torch.float32, device=dev)
Y = torch.zeros(P, N, dtype=torch.bfloat16, device=dev)
svh = torch.full((E * N,), 0.5, dtype=torch.float16, device=dev)
wts = torch.full((P,), 1.0 / SLOTS, dtype=torch.float32, device=dev)

counts = torch.zeros(E, dtype=torch.int32, device=dev)
uids = torch.zeros(MAXU, dtype=torch.int32, device=dev)
ucount = torch.zeros(1, dtype=torch.int32, device=dev)
members = torch.full((MAXU, MAXM), -1, dtype=torch.int32, device=dev)
work_d = torch.zeros(ND * 2, dtype=torch.int32, device=dev)

torch.manual_seed(1234)
pick6 = torch.argsort(torch.rand(R, 384, device=dev), dim=1)[:, :6].to(torch.int32)
pick = torch.full((R, SLOTS), 384, dtype=torch.int32, device=dev)
pick[:, :6] = pick6
pick = pick.contiguous()
m.group_count(pick, counts, R, SLOTS, E)
m.group_place(pick, counts, uids, ucount, members, R, SLOTS, E)
m.m2_work(counts, uids, ucount, work_d, 32)
torch.cuda.synchronize()

nu = int(ucount.item())
live = counts[uids[:nu].to(torch.int64)].to(torch.int64)
print(f"down 生产形状 K={K} N={N} P={P}  ucount={nu}  活跃组={int(((live+31)//32).sum())}/{ND}")
print(f"v1 程序数 {N//128*ND}   down3 程序数 {32*(N//256)*MAXU}")
print(f"显存 {torch.cuda.memory_allocated()/2**30:.2f} GiB")


def run_v1():
    m.grouped_rows(xd, xd, ptrs, ptrs, k2, k2, uids, ucount, members, Z,
                   1, K, N, P, 1, SLOTS, 2, 8, 4, 1, 2, K2, K2, 0, work_d)


def run_down3():
    m.grouped_down3(xd, ptrs, k2, uids, ucount, members, svh, wts, Y,
                    K, N, P, SLOTS, 2, K2, K2)


ARMS = {"v1": run_v1, "down3": run_down3}

run_v1(); torch.cuda.synchronize()
assert Z.abs().max().item() > 0, "v1 输出全零"
run_down3(); torch.cuda.synchronize()
ymax = Y.abs().max().item()
assert ymax > 0 and torch.isfinite(Y.float()).all(), "down3 输出全零或含非有限值"
print(f"非空校验：v1 max|Z|={Z.abs().max().item():.4f}  down3 max|Y|={ymax:.4f}"
      f"（两者输出语义不同：Z 未乘权重/scale，Y 已融合 epilogue，故不做数值等价断言）")


def timeit(fn, n):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts)//2], ts[0], ts[-1]


res = {"K": K, "N": N, "P": P, "arms": []}
if args.only:
    med, lo, hi = timeit(ARMS[args.only], 2 if args.ncu_warm else args.reps)
    print(f"{args.only:6s} 中位 {med:.4f} ms  min {lo:.4f} max {hi:.4f}")
    res["arms"].append({"arm": args.only, "median_ms": med})
else:
    for i, name in enumerate(args.arms.split(",")):
        med, lo, hi = timeit(ARMS[name], args.reps)
        print(f"臂{i+1} {name:6s} 中位 {med:.4f} ms  min {lo:.4f} max {hi:.4f}")
        res["arms"].append({"arm": name, "median_ms": med, "min_ms": lo, "max_ms": hi})
    v = [a["median_ms"] for a in res["arms"] if a["arm"] == "v1"]
    d3 = [a["median_ms"] for a in res["arms"] if a["arm"] == "down3"]
    if v and d3:
        print(f"v1 极差 {max(v)-min(v):.4f} ms ({100*(max(v)-min(v))/min(v):.2f}%)")
        pr = list(zip(v, d3))
        for j, (x, y) in enumerate(pr):
            print(f"  配对{j+1}: v1 {x:.4f} → down3 {y:.4f}  {100*(y-x)/x:+.2f}%")
        res["paired_mean_pct"] = sum(100*(y-x)/x for x, y in pr)/len(pr)
        print(f"配对均值 {res['paired_mean_pct']:+.2f}%")
    print("生产靶（trace 满块中位）6.6802 ms")

if args.out:
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("WROTE", args.out)
print("PROBE_OK")