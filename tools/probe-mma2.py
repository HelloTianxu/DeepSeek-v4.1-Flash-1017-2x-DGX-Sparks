#!/usr/bin/env python3
"""E-SELF-08：v1（权重走寄存器中转）vs mma2（权重也走 cp.async）同场 A/B。

判决目标：E-SELF-07 把卡口钉在「寄存器同时限制驻留 warp 数与承载权重暂存」。
mma2 用 S=4 级 cp.async 流权重，shared 61440 B ⇒ 1 block/SM（占用率比 v1 的 2 blocks 更低）。
  · 若 mma2 在更低占用率下达成带宽更高 ⇒ 卡点是「在飞字节数」，cp.async 是杠杆，木桶分析成立。
  · 若更低 ⇒ 驻留 warp 数主导，需另找杠杆。

形状说明：mma2 要求 N % M2_COLS(256) == 0，我方 TP2 本地 I=1152 不是 256 的倍数，
故**两臂都用全宽 N=2304**（v1 只要 N%64==0，也满足），同 K=5120、同 P=14336、同权重字节。
这不是生产形状，但两臂完全同形，A/B 有效。

位一致提醒：源码 :800 注释明示 mma2 「not grouped_kernel's split order (other bits than
the decode windows')」⇒ **与 v1 非位一致**，本脚本做数值接近性校验以防测到跳过工作的内核。
"""
import os, sys, json, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2304, help="gate/up 的 N（须为 256 的倍数才能跑 mma2）")
ap.add_argument("--reps", type=int, default=8)
ap.add_argument("--arms", default="v1,mma2,v1,mma2", help="交替臂序，小效应看配对差")
ap.add_argument("--out", default="")
ap.add_argument("--ncu-warm", action="store_true")
ap.add_argument("--only", default="", help="只跑一个臂（供 ncu 用）：v1 | mma2")
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
D = 5120
N = args.n
E = 385
MAXM = 2048
MAXU = min(P, E)
K2 = 6
TW = 4 * K2
assert N % 64 == 0, "v1 需要 N%64==0"


def list_len(p, maxu, rows):
    return (p + maxu * (rows - 1)) // rows


NGU = list_len(P, MAXU, 64)

# 权重：每专家每投影 KT*NTILES*TW words
WORDS = (D // 16) * (N // 16) * TW
STRIDE_B = WORDS * 4


def make_proj():
    buf = torch.randint(-2**31, 2**31 - 1, (E * WORDS,), dtype=torch.int32, device=dev)
    ptrs = (buf.data_ptr() + torch.arange(E, dtype=torch.int64, device=dev) * STRIDE_B).contiguous()
    k2 = torch.full((E,), K2, dtype=torch.int32, device=dev)
    return buf, ptrs, k2


gate_b, gate_p, gate_k2 = make_proj()
up_b, up_p, up_k2 = make_proj()
xg = torch.randn(P, D, dtype=torch.float16, device=dev)
xu = torch.randn(P, D, dtype=torch.float16, device=dev)
Z = torch.zeros(2 * P * N, dtype=torch.float32, device=dev)

counts = torch.zeros(E, dtype=torch.int32, device=dev)
uids = torch.zeros(MAXU, dtype=torch.int32, device=dev)
ucount = torch.zeros(1, dtype=torch.int32, device=dev)
members = torch.full((MAXU, MAXM), -1, dtype=torch.int32, device=dev)
work_gu = torch.zeros(NGU * 2, dtype=torch.int32, device=dev)

torch.manual_seed(1234)
pick6 = torch.argsort(torch.rand(R, 384, device=dev), dim=1)[:, :6].to(torch.int32)
pick = torch.full((R, SLOTS), 384, dtype=torch.int32, device=dev)
pick[:, :6] = pick6
pick = pick.contiguous()

m.group_count(pick, counts, R, SLOTS, E)
m.group_place(pick, counts, uids, ucount, members, R, SLOTS, E)
m.m2_work(counts, uids, ucount, work_gu, 64)
torch.cuda.synchronize()

nu = int(ucount.item())
live = counts[uids[:nu].to(torch.int64)].to(torch.int64)
live_gu = int(((live + 63) // 64).sum())
print(f"N={N}  P={P}  K={D}  ucount={nu}  活跃组={live_gu}/{NGU}")
print(f"权重字节/专家/投影 = {STRIDE_B/2**20:.3f} MiB   两投影合计 {2*E*STRIDE_B/2**30:.3f} GiB")
print(f"显存 {torch.cuda.memory_allocated()/2**30:.2f} GiB")


def run_v1():
    m.grouped_mma(xg, xu, gate_p, up_p, gate_k2, up_k2, uids, ucount, members, Z,
                  2, D, N, P, 4, SLOTS, 2, 4, K2, K2, 1, work_gu)


def run_mma2():
    m.grouped_mma2(xg, xu, gate_p, up_p, gate_k2, up_k2, uids, ucount, members, Z,
                   2, D, N, P, SLOTS, 2, K2, K2, 64, work_gu)


ARMS = {"v1": run_v1, "mma2": run_mma2}

# 数值接近性校验：两臂都真的算了同一件事（非位一致，但应接近）
run_v1(); torch.cuda.synchronize()
z1 = Z[:2 * P * N].clone()
run_mma2(); torch.cuda.synchronize()
z2 = Z[:2 * P * N].clone()
den = z1.abs().max().item()
diff = (z1 - z2).abs().max().item()
rel = diff / den if den else 0
print(f"数值校验：max|v1|={den:.4f}  max|v1-mma2|={diff:.6f}  相对 {rel:.3e}")
assert den > 0, "v1 输出全零，测到空内核"
assert torch.isfinite(z2).all(), "mma2 输出含非有限值"
assert rel < 0.05, f"两臂输出差异过大({rel:.3e})，可能有一臂未做同样工作"
print("数值校验通过（非位一致，属预期：源码 :800 明示 split order 不同）")


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
    return ts[len(ts) // 2], ts[0], ts[-1]


res = {"N": N, "P": P, "K": D, "live_gu": live_gu, "rel_diff": rel,
       "mem_GiB": torch.cuda.memory_allocated() / 2**30, "arms": []}

if args.only:
    fn = ARMS[args.only]
    med, lo, hi = timeit(fn, 2 if args.ncu_warm else args.reps)
    print(f"{args.only:5s} 中位 {med:.4f} ms  min {lo:.4f} max {hi:.4f}")
    res["arms"].append({"arm": args.only, "median_ms": med})
else:
    seq = args.arms.split(",")
    for i, name in enumerate(seq):
        med, lo, hi = timeit(ARMS[name], args.reps)
        print(f"臂{i+1} {name:5s} 中位 {med:.4f} ms  min {lo:.4f} max {hi:.4f}")
        res["arms"].append({"arm": name, "median_ms": med, "min_ms": lo, "max_ms": hi})
    v1s = [a["median_ms"] for a in res["arms"] if a["arm"] == "v1"]
    m2s = [a["median_ms"] for a in res["arms"] if a["arm"] == "mma2"]
    if v1s and m2s:
        print(f"v1 极差 {max(v1s)-min(v1s):.4f} ms ({100*(max(v1s)-min(v1s))/min(v1s):.2f}%)")
        pairs = list(zip(v1s, m2s))
        for j, (a, b) in enumerate(pairs):
            print(f"  配对{j+1}: v1 {a:.4f} → mma2 {b:.4f}  差 {b-a:+.4f} ms ({100*(b-a)/a:+.2f}%)")
        avg = sum(100 * (b - a) / a for a, b in pairs) / len(pairs)
        print(f"配对均值 {avg:+.2f}%   mma2/v1 时长比 {sum(m2s)/len(m2s)/(sum(v1s)/len(v1s)):.3f}")
        res["paired_mean_pct"] = avg

if args.out:
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("WROTE", args.out)
print("PROBE_OK")