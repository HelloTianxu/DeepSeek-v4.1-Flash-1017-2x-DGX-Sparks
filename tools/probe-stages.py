#!/usr/bin/env python3
"""E-SELF-09：把 mma2 的 cp.async 流水深度 S 推到底，单因子扫在飞字节数。

S 只改「暂存环有几级」，不改形状、不改权重字节、不改累加顺序、不改占用率（本来就 1 block/SM）。
在飞组数 = S-2 ⇒ S∈{3,4,5,6} 对应 {1,2,3,4} 组，S=6 把在飞字节相对基线 S=4 翻倍。
硬件闸：GB10 optin 101376 B，每级 15360 B ⇒ S=7 (107520 B) 被拒，已在源码 TORCH_CHECK 拦住。

正确性闸（比 E-SELF-08 更强）：源码注释明示每个输出是「一条按 K 顺序的 fp32 mma 链，
与窗口无关」⇒ **改 S 必须位一致**。本脚本对每个 S 断言 torch.equal(Z_S, Z_S4)。

用法：
  probe-stages.py --build-only                 # 只编译（-Xptxas -v 抓 REG/spill 做编译期预筛）
  probe-stages.py --stages 3,5,6 --reps 8      # 每个 S 与同场 S=4 做 4,S,4,S 交替
  probe-stages.py --only 6 --ncu-warm          # 供 ncu 采集
"""
import os, json, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2304)
ap.add_argument("--reps", type=int, default=8)
ap.add_argument("--stages", default="3,5,6")
ap.add_argument("--only", type=int, default=0)
ap.add_argument("--ncu-warm", action="store_true")
ap.add_argument("--build-only", action="store_true")
ap.add_argument("--out", default="")
args = ap.parse_args()

import torch
from torch.utils.cpp_extension import load

SRC = os.path.expanduser("~/ptxwork/exl3src-s")
BUILD = os.path.expanduser("~/ptxwork/probe-build-s")
os.makedirs(BUILD, exist_ok=True)
import time
t0 = time.time()
m = load(name="tf_exl3_probe_s",
         sources=[os.path.join(SRC, n) for n in
                  ["experts.cpp", "experts.cu", "experts_cb0.cu", "experts_cb1.cu", "experts_cb2.cu"]],
         extra_cflags=["-O3"],
         extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v",
                            "-gencode=arch=compute_121,code=sm_121"],
         build_directory=BUILD, verbose=True)
print(f"BUILD_SECONDS {time.time()-t0:.1f}")
assert hasattr(m, "grouped_mma2"), "导出缺失"
if args.build_only:
    print("BUILD_OK")
    raise SystemExit(0)

dev = "cuda"
R, SLOTS = 2048, 7
P = R * SLOTS
K, N = 5120, args.n
E, MAXM = 385, 2048
MAXU = min(P, E)
K2 = 6
TW = 4 * K2
assert N % 256 == 0 and K % 64 == 0

WORDS = (K // 16) * (N // 16) * TW
buf = torch.randint(-2**31, 2**31 - 1, (2 * E * WORDS,), dtype=torch.int32, device=dev)
gptr = (buf.data_ptr() + torch.arange(E, dtype=torch.int64, device=dev) * WORDS * 4).contiguous()
uptr = (buf.data_ptr() + (E + torch.arange(E, dtype=torch.int64, device=dev)) * WORDS * 4).contiguous()
k2 = torch.full((E,), K2, dtype=torch.int32, device=dev)
xg = torch.randn(P, K, dtype=torch.float16, device=dev)
xu = torch.randn(P, K, dtype=torch.float16, device=dev)
Z = torch.zeros(2 * P * N, dtype=torch.float32, device=dev)

counts = torch.zeros(E, dtype=torch.int32, device=dev)
uids = torch.zeros(MAXU, dtype=torch.int32, device=dev)
ucount = torch.zeros(1, dtype=torch.int32, device=dev)
members = torch.full((MAXU, MAXM), -1, dtype=torch.int32, device=dev)
NG = (P + MAXU * 63) // 64
work = torch.zeros(NG * 2, dtype=torch.int32, device=dev)

torch.manual_seed(1234)
pick6 = torch.argsort(torch.rand(R, 384, device=dev), dim=1)[:, :6].to(torch.int32)
pick = torch.full((R, SLOTS), 384, dtype=torch.int32, device=dev)
pick[:, :6] = pick6
pick = pick.contiguous()
m.group_count(pick, counts, R, SLOTS, E)
m.group_place(pick, counts, uids, ucount, members, R, SLOTS, E)
m.m2_work(counts, uids, ucount, work, 64)
torch.cuda.synchronize()

nu = int(ucount.item())
live = counts[uids[:nu].to(torch.int64)].to(torch.int64)
print(f"N={N} K={K} P={P} ucount={nu} 活跃组={int(((live+63)//64).sum())}/{NG}")
print(f"显存 {torch.cuda.memory_allocated()/2**30:.2f} GiB")


def run(s):
    os.environ["TF_PROBE_M2_S"] = str(s)
    m.grouped_mma2(xg, xu, gptr, uptr, k2, k2, uids, ucount, members, Z,
                   K, N, P, SLOTS, 2, K2, K2, 64, work)


# ---- 位一致闸 ----
run(4); torch.cuda.synchronize()
ref = Z.clone()
assert ref.abs().max().item() > 0 and torch.isfinite(ref).all(), "S=4 参考输出全零或非有限"
print(f"S=4 参考 max|Z|={ref.abs().max().item():.4f}")
for s in [3, 5, 6]:
    Z.zero_(); run(s); torch.cuda.synchronize()
    same = torch.equal(Z, ref)
    md = (Z - ref).abs().max().item()
    print(f"  S={s} 位一致={same}  max|差|={md:.6g}")
    assert same, f"S={s} 与 S=4 非位一致 —— 环深度不该改变累加顺序，停止"
print("位一致闸通过：S 只改暂存，不改结果")


def timeit(s, n):
    for _ in range(3):
        run(s)
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); run(s); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts)//2], ts[0], ts[-1]


res = {"N": N, "K": K, "P": P, "stages": []}
if args.only:
    med, lo, hi = timeit(args.only, 2 if args.ncu_warm else args.reps)
    print(f"S={args.only} 中位 {med:.4f} ms  min {lo:.4f} max {hi:.4f}")
    res["stages"].append({"S": args.only, "median_ms": med})
else:
    for s in [int(x) for x in args.stages.split(",")]:
        pair = []
        for arm in (4, s, 4, s):
            med, lo, hi = timeit(arm, args.reps)
            pair.append(med)
            print(f"  S={arm} 中位 {med:.4f} ms  min {lo:.4f} max {hi:.4f}")
        d1 = 100*(pair[1]-pair[0])/pair[0]
        d2 = 100*(pair[3]-pair[2])/pair[2]
        print(f"S={s} vs S=4 配对: {d1:+.2f}% / {d2:+.2f}%  均值 {(d1+d2)/2:+.2f}%"
              f"  同向={'是' if d1*d2>0 else '否'}")
        res["stages"].append({"S": s, "control_ms": [pair[0], pair[2]],
                              "arm_ms": [pair[1], pair[3]],
                              "paired_pct": [d1, d2], "mean_pct": (d1+d2)/2})
    print("\n汇总（负=更快）：")
    for r in res["stages"]:
        print(f"  S={r['S']}  {r['mean_pct']:+.2f}%")

if args.out:
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("WROTE", args.out)
print("PROBE_OK")