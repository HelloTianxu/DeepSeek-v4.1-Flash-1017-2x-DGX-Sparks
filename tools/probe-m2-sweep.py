#!/usr/bin/env python3
"""E-SELF-09：把 cp.async 级数 S 与每 SM block 数推到底，榨在飞字节。

E-SELF-08 已证卡口是片内递货。本轮沿同一杠杆做二维扫描：
  S    ∈ {3,4,5,6}  —— 每 block 的 cp.async 级数，直接决定在飞字节数
  MINB ∈ {1,2}      —— __launch_bounds__ 最小 block 数，压寄存器换驻留

shared(S) = S*64*72*2 + S*4*16*4*6*4 = S*15360 B
  S=3 → 46080（≤51200 ⇒ 若 REG≤128 可 2 blocks）
  S=4 → 61440（>51200 ⇒ 只能 1 block，即上游默认）
  S=5 → 76800，S=6 → 92160（均 ≤102400 optin 上限，1 block）
编译期实测：S 各档 REG 均 133（⇒1 block）；MINB=2 档 REG 126（⇒2 blocks 若 shared 允许）。
全部 STACK:0 LOCAL:0，无 spill。

臂用 "S:MINB" 表示，"0:1" = 上游默认（模板 S_=0 走 R>M2_ROWS?3:4 ⇒ S=4）。
环境变量在**每次 launch 时**被 host 读取，故同进程内可切臂，A/B 同场。

正确性闸：S 只改流水深度、不改 k 累加顺序 ⇒ 各臂输出应与默认**位一致**。
若不一致，说明改动动了语义，立即报警（而不是把结果当性能数据用）。
"""
import os, json, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2304)
ap.add_argument("--reps", type=int, default=8)
ap.add_argument("--arms", default="0:1,3:1,0:1,5:1,0:1,6:1,0:1,3:2,0:1,4:2")
ap.add_argument("--arm", default="", help="只跑一个臂（供 ncu 用），如 3:2")
ap.add_argument("--ncu-warm", action="store_true")
ap.add_argument("--out", default="")
ap.add_argument("--no-check", action="store_true", help="跳过位一致闸（ncu 单臂时用）")
args = ap.parse_args()

import torch
from torch.utils.cpp_extension import load

BUILD = os.path.expanduser("~/ptxwork/probe-build-sweep")
SRC = os.path.expanduser("~/ptxwork/exl3src-sweep")
m = load(name="tf_exl3_sweep",
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
E, MAXM = 385, 2048
MAXU = min(P, E)
K2 = 6
TW = 4 * K2
NGU = (P + MAXU * 63) // 64
WORDS = (D // 16) * (N // 16) * TW
STRIDE_B = WORDS * 4


def make_proj():
    buf = torch.randint(-2**31, 2**31 - 1, (E * WORDS,), dtype=torch.int32, device=dev)
    ptrs = (buf.data_ptr() + torch.arange(E, dtype=torch.int64, device=dev) * STRIDE_B).contiguous()
    return buf, ptrs, torch.full((E,), K2, dtype=torch.int32, device=dev)


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
print(f"N={N} P={P} K={D} ucount={nu} 活跃组={int(((live+63)//64).sum())}/{NGU}")
print(f"显存 {torch.cuda.memory_allocated()/2**30:.2f} GiB")


def set_arm(a):
    # "S:MINB" 或 "S:MINB:KB"；KB 默认 4
    parts = a.split(":")
    s, b = parts[0], parts[1]
    kb = parts[2] if len(parts) > 2 else "4"
    os.environ["TF_PROBE_M2_S"] = s
    os.environ["TF_PROBE_M2_MINB"] = b
    os.environ["TF_PROBE_M2_KB"] = kb
    os.environ["TF_PROBE_M2_MODE"] = "0"


def run():
    m.grouped_mma2(xg, xu, gate_p, up_p, gate_k2, up_k2, uids, ucount, members, Z,
                   2, D, N, P, SLOTS, 2, K2, K2, 64, work_gu)


def smem_of(a):
    parts = a.split(":")
    kb = int(parts[2]) if len(parts) > 2 else 4
    s = int(parts[0]) or (3 if kb == 8 else 4)
    return s * 64 * (kb * 16 + 8) * 2 + s * kb * 16 * 4 * 6 * 4


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


# ---- 位一致闸：各臂应与上游默认逐位相同 ----
uniq = []
for a in args.arms.split(","):
    if a not in uniq:
        uniq.append(a)
if args.arm:
    uniq = [args.arm]

ref = None
if not args.no_check:
    set_arm("0:1"); run(); torch.cuda.synchronize()
    ref = Z.clone()
    assert ref.abs().max().item() > 0, "默认臂输出全零"
    print(f"基准 max|Z|={ref.abs().max().item():.4f}")
    for a in uniq:
        set_arm(a); run(); torch.cuda.synchronize()
        same = torch.equal(Z, ref)
        md = (Z - ref).abs().max().item()
        print(f"  臂 {a:5s} shared={smem_of(a):6d}B  位一致={same}  maxdiff={md:.6g}")
        assert torch.isfinite(Z).all(), f"臂 {a} 输出含非有限值"
        if not same:
            print(f"  ⚠️ 臂 {a} 与默认非位一致 —— S/MINB 本不该改数学，需查明再采信性能数")
    print("位一致闸完成")

res = {"N": N, "P": P, "K": D, "arms": []}
if args.arm:
    set_arm(args.arm)
    med, lo, hi = timeit(2 if args.ncu_warm else args.reps)
    print(f"臂 {args.arm:5s} 中位 {med:.4f} ms  min {lo:.4f} max {hi:.4f}")
    res["arms"].append({"arm": args.arm, "median_ms": med, "smem": smem_of(args.arm)})
else:
    for i, a in enumerate(args.arms.split(",")):
        set_arm(a)
        med, lo, hi = timeit(args.reps)
        print(f"臂{i+1:2d} {a:5s} shared={smem_of(a):6d}B 中位 {med:.4f} ms  min {lo:.4f} max {hi:.4f}")
        res["arms"].append({"arm": a, "median_ms": med, "min_ms": lo, "max_ms": hi,
                            "smem": smem_of(a)})
    seq = [x["arm"] for x in res["arms"]]
    base = [x["median_ms"] for x in res["arms"] if x["arm"] == "0:1"]
    if base:
        print(f"\n默认臂(0:1) 极差 {max(base)-min(base):.4f} ms ({100*(max(base)-min(base))/min(base):.2f}%)"
              f"  中位 {sorted(base)[len(base)//2]:.4f} ms")
        bmed = sorted(base)[len(base) // 2]
        print(f"{'臂':>6}{'中位ms':>10}{'相对默认':>10}{'相邻默认配对':>14}")
        for x in res["arms"]:
            if x["arm"] == "0:1":
                continue
            # 与该臂紧邻的前一个默认臂配对，避免用全局均值掩盖漂移
            i = seq.index(x["arm"])
            prev = next((res["arms"][j]["median_ms"] for j in range(i - 1, -1, -1)
                         if res["arms"][j]["arm"] == "0:1"), bmed)
            print(f"{x['arm']:>6}{x['median_ms']:>10.4f}{100*(x['median_ms']-bmed)/bmed:>9.2f}%"
                  f"{100*(x['median_ms']-prev)/prev:>13.2f}%")
            x["vs_adjacent_base_pct"] = 100 * (x["median_ms"] - prev) / prev
            x["vs_base_median_pct"] = 100 * (x["median_ms"] - bmed) / bmed

if args.out:
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("WROTE", args.out)
print("PROBE_OK")