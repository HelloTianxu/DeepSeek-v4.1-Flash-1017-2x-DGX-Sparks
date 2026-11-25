#!/usr/bin/env python3
"""E-BANK-01 实验A：down 内核 red padding 的同场 A/B + 位一致闸。

假设：red row stride = NT*16 = 128 float 是 32 banks 的整数倍
      ⇒ 一个 warp 内 lane%4 相同的 8 个线程（g8=lane>>2 = 0..7）落同一 bank ⇒ 8 路 store 冲突。
      实测 bank_conflicts_st / shared_st 指令 = 6.02×（E-SELF-07 pipe-rows.csv）。
干预：stride 128 → 129（PAD=1），只改布局不改数值与求和顺序 ⇒ 应位一致。

形状与 probe-down3.py 完全一致（生产 down：K=I=1152、N=D=5120、P=14336、nt=8、warps=4、pf=1、g=2）。
**注意 lo/hi 传 K2=6 ⇒ dispatch 走 <...,2,10,...> 分支，与生产台账记的 <...,2,16,...> 不同；
   但 red 布局只由 NT/W 决定，两者一致 ⇒ bank 冲突机制相同。此差异已标注，不外推绝对时长。**

同场：同一进程、同一模块、同一输入，只翻 TF_EXL3_RED_PAD（host 每次 launch 读 getenv，非 static）。
用法：python3 probe-red-pad.py [--reps 8] [--out res.json]
"""
import os, json, argparse, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=8)
ap.add_argument("--out", default="")
ap.add_argument("--only", default="")          # "p0"/"p1"：给 ncu 单臂用
ap.add_argument("--ncu-warm", action="store_true")
args = ap.parse_args()

SRC = os.path.expanduser("~/ptxwork/exl3src-pad")
BUILD = os.path.expanduser("~/ptxwork/probe-build-pad")
os.makedirs(BUILD, exist_ok=True)
os.environ.setdefault("MAX_JOBS", "4")

import torch
from torch.utils.cpp_extension import load

t0 = time.time()
m = load(name="tf_exl3_probe_pad",
         sources=[os.path.join(SRC, n) for n in
                  ["experts.cpp", "experts.cu", "experts_cb0.cu", "experts_cb1.cu", "experts_cb2.cu"]],
         extra_cflags=["-O3"],
         extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121,code=sm_121"],
         build_directory=BUILD, verbose=False)
print(f"BUILD_OK {time.time()-t0:.1f}s", flush=True)

dev = "cuda"
R, SLOTS = 2048, 7
P = R * SLOTS
K, N = 1152, 5120
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
print(f"形状 K={K} N={N} P={P} ucount={int(ucount.item())}  显存 {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)


def run(pad):
    os.environ["TF_EXL3_RED_PAD"] = "1" if pad else "0"
    m.grouped_rows(xd, xd, ptrs, ptrs, k2, k2, uids, ucount, members, Z,
                   1, K, N, P, 1, SLOTS, 2, 8, 4, 1, 2, K2, K2, 0, work_d)


# ---- 位一致闸（同一输入，两臂各自覆盖写 Z）----
Z.zero_(); run(0); torch.cuda.synchronize()
z0 = Z.clone()
assert z0.abs().max().item() > 0, "p0 输出全零"
Z.zero_(); run(1); torch.cuda.synchronize()
z1 = Z.clone()
bit_equal = torch.equal(z0, z1)
maxdiff = (z0 - z1).abs().max().item()
print(f"位一致闸: torch.equal={bit_equal}  maxdiff={maxdiff:.6e}  max|Z|={z0.abs().max().item():.4f}", flush=True)
if not bit_equal:
    print("BIT_IDENTITY_FAILED —— padding 改变了数值，不可上线", flush=True)

def timed(pad, reps):
    run(pad); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        a = time.perf_counter()
        run(pad)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - a) * 1e3)
    return ts

res = {"K": K, "N": N, "P": P, "bit_equal": bool(bit_equal), "maxdiff": maxdiff, "arms": []}
if args.only:
    pad = 1 if args.only == "p1" else 0
    n = 6 if args.ncu_warm else 1
    for _ in range(n):
        run(pad)
    torch.cuda.synchronize()
    print(f"ONLY {args.only} done", flush=True)
else:
    # 交替 A/B/A/B...，每臂 reps 次，看配对差是否同向
    seq = [0, 1] * (args.reps // 2)
    for pad in seq:
        ts = timed(pad, 5)
        med = statistics.median(ts)
        res["arms"].append({"arm": f"p{pad}", "median_ms": med, "min_ms": min(ts), "max_ms": max(ts)})
        print(f"  p{pad}  median {med:.4f} ms  [{min(ts):.4f}, {max(ts):.4f}]", flush=True)
    p0 = [a["median_ms"] for a in res["arms"] if a["arm"] == "p0"]
    p1 = [a["median_ms"] for a in res["arms"] if a["arm"] == "p1"]
    pairs = []
    print(f"p0 极差 {max(p0)-min(p0):.4f} ms ({100*(max(p0)-min(p0))/min(p0):.2f}%)")
    for j, (x, y) in enumerate(zip(p0, p1)):
        d = 100 * (y - x) / x
        pairs.append(d)
        print(f"  配对{j+1}: p0 {x:.4f} → p1 {y:.4f}  {d:+.3f}%")
    res["paired_pct"] = pairs
    res["mean_paired_pct"] = sum(pairs) / len(pairs)
    res["all_same_sign"] = all(d < 0 for d in pairs) or all(d > 0 for d in pairs)
    print(f"均值配对 {res['mean_paired_pct']:+.3f}%  同向={res['all_same_sign']}", flush=True)

if args.out:
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"WROTE {args.out}", flush=True)