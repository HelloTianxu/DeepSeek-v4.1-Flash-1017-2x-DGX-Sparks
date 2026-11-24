#!/usr/bin/env python3
"""E-SELF-09d：分离「访存 / 解码 / MMA」三者谁是串行加数。

MODE 0 = 完整；MODE 1 = 留解码去 MMA；MODE 2 = 留 MMA 去解码。
纯读屋顶（probe-pattern-roof，同形状同字节）= 14.5 ms / 253.7 GB/s。
若 MODE1 ≈ 纯读 ⇒ 解码基本被藏住，MMA 是加数；若 MODE1 ≈ 完整 ⇒ 解码是加数。
MODE1/2 的**输出无意义**（故意少算），只看时长与计数器，故跳过数值闸。
"""
import os, json, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2304)
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--modes", default="0,4,0,4,0,5,0,5")
ap.add_argument("--mode", default="")
ap.add_argument("--ncu-warm", action="store_true")
ap.add_argument("--out", default="")
args = ap.parse_args()

import torch
from torch.utils.cpp_extension import load

m = load(name="tf_exl3_sweep",
         sources=[os.path.join(os.path.expanduser("~/ptxwork/exl3src-sweep"), n) for n in
                  ["experts.cpp", "experts.cu", "experts_cb0.cu", "experts_cb1.cu", "experts_cb2.cu"]],
         extra_cflags=["-O3"],
         extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121,code=sm_121"],
         build_directory=os.path.expanduser("~/ptxwork/probe-build-sweep"), verbose=False)

dev = "cuda"
R, SLOTS = 2048, 7
P = R * SLOTS
D, N = 5120, args.n
E, MAXM = 385, 2048
MAXU = min(P, E)
K2, TW = 6, 24
NGU = (P + MAXU * 63) // 64
WORDS = (D // 16) * (N // 16) * TW
STRIDE_B = WORDS * 4


def mk():
    b = torch.randint(-2**31, 2**31 - 1, (E * WORDS,), dtype=torch.int32, device=dev)
    p = (b.data_ptr() + torch.arange(E, dtype=torch.int64, device=dev) * STRIDE_B).contiguous()
    return b, p, torch.full((E,), K2, dtype=torch.int32, device=dev)


gb, gp, gk = mk()
ub, up, uk = mk()
xg = torch.randn(P, D, dtype=torch.float16, device=dev)
xu = torch.randn(P, D, dtype=torch.float16, device=dev)
Z = torch.zeros(2 * P * N, dtype=torch.float32, device=dev)
counts = torch.zeros(E, dtype=torch.int32, device=dev)
uids = torch.zeros(MAXU, dtype=torch.int32, device=dev)
ucount = torch.zeros(1, dtype=torch.int32, device=dev)
members = torch.full((MAXU, MAXM), -1, dtype=torch.int32, device=dev)
work = torch.zeros(NGU * 2, dtype=torch.int32, device=dev)
torch.manual_seed(1234)
p6 = torch.argsort(torch.rand(R, 384, device=dev), dim=1)[:, :6].to(torch.int32)
pick = torch.full((R, SLOTS), 384, dtype=torch.int32, device=dev)
pick[:, :6] = p6
pick = pick.contiguous()
m.group_count(pick, counts, R, SLOTS, E)
m.group_place(pick, counts, uids, ucount, members, R, SLOTS, E)
m.m2_work(counts, uids, ucount, work, 64)
torch.cuda.synchronize()
live = counts[uids[:int(ucount.item())].to(torch.int64)].to(torch.int64)
print(f"N={N} P={P} K={D} 活跃组={int(((live+63)//64).sum())}/{NGU}  显存 {torch.cuda.memory_allocated()/2**30:.2f} GiB")


def run():
    m.grouped_mma2(xg, xu, gp, up, gk, uk, uids, ucount, members, Z,
                   2, D, N, P, SLOTS, 2, K2, K2, 64, work)


def setmode(v):
    os.environ["TF_PROBE_M2_S"] = "0"
    os.environ["TF_PROBE_M2_MINB"] = "1"
    os.environ["TF_PROBE_M2_MODE"] = str(v)


def timeit(n):
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); run(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ts.sort()
    return ts[len(ts) // 2], ts[0], ts[-1]


NAME = {"0": "完整", "1": "去MMA(留解码)", "2": "去解码(留MMA)", "3": "纯plumbing(去解码+MMA)",
        "4": "无栅栏(仅计时)", "5": "无栅栏无wait(仅计时)"}
res = {"N": N, "runs": []}
seq = [args.mode] if args.mode else args.modes.split(",")
for i, v in enumerate(seq):
    setmode(v)
    med, lo, hi = timeit(2 if args.ncu_warm else args.reps)
    print(f"臂{i+1:2d} MODE={v} {NAME[v]:<16} 中位 {med:.4f} ms  min {lo:.4f} max {hi:.4f}")
    res["runs"].append({"mode": v, "median_ms": med, "min_ms": lo, "max_ms": hi})

if not args.mode:
    print()
    for v in ("0", "1", "2", "3", "4", "5"):
        xs = [x["min_ms"] for x in res["runs"] if x["mode"] == v]
        if xs:
            print(f"MODE={v} {NAME[v]:<16} min-of-min {min(xs):.4f} ms  样本 {len(xs)}")
    b0 = min(x["min_ms"] for x in res["runs"] if x["mode"] == "0")
    for v in ("1", "2", "3", "4", "5"):
        xs = [x["min_ms"] for x in res["runs"] if x["mode"] == v]
        if xs:
            print(f"  去掉{ {'1':'MMA','2':'解码','3':'解码+MMA','4':'block栅栏','5':'栅栏+wait'}[v] }后 {min(xs):.4f} ms ⇒ 串行贡献 {b0-min(xs):+.4f} ms"
                  f" ({100*(b0-min(xs))/b0:.1f}%)")
    print("参照：同形状纯读屋顶 14.5 ms（253.7 GB/s）")

if args.out:
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("WROTE", args.out)
print("PROBE_OK")