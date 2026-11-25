#!/usr/bin/env python3
"""E-SIM-02：把仿真 S3/S4 落地——S=2 + KB=2 + MINB=3 是否真能无重罚拿到 3 blocks。

编译期预筛（cuobjdump -res-usage 实读，HI=10 生产同款 range）：
  S=3 KB=4 MINB=1: REG 124 spill  0B shared 45312 ⇒ blocks min(2,2)=2  ← 生产
  S=2 KB=4 MINB=1: REG 106 spill  0B shared 36096 ⇒ blocks min(2,2)=2
  S=2 KB=2 MINB=1: REG  97 spill  0B shared 19712 ⇒ blocks min(2,5)=2
  S=2 KB=4 MINB=3: REG  80 spill 32B shared 36096 ⇒ blocks min(3,2)=2  白付 spill
  S=2 KB=2 MINB=3: REG  80 spill  8B shared 19712 ⇒ blocks min(3,5)=3  ← 唯一可达且 spill 最小
对照臂同场重测（禁止复用历史基线），A/B/A/B 交替，臂间 0.25s 冷却。
用法：python3 probe-s2.py [--reps 12] [--out s2.json] [--only new|base]
"""
import os, json, argparse, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=12)
ap.add_argument("--out", default="")
ap.add_argument("--only", default="")   # "mb,kb" 例 "3,2"：给 ncu 单臂用
args = ap.parse_args()

BUILD_ROOT = os.path.expanduser("~/ptxwork")
os.environ.setdefault("MAX_JOBS", "2")
import torch
from torch.utils.cpp_extension import load

mods = {}
TREES = {"base": ("exl3src-mb1", "probe-build-mb1", 1),
         "new":  ("exl3src-s2",  "probe-build-s2mb3", 3)}
for tag, (sd, bdn, mb) in TREES.items():
    src = os.path.join(BUILD_ROOT, sd)
    bd = os.path.join(BUILD_ROOT, bdn)
    os.makedirs(bd, exist_ok=True)
    t0 = time.time()
    mods[tag] = load(name=f"tf_exl3_{bdn.replace(chr(45),chr(95))}",
                    sources=[os.path.join(src, n) for n in
                             ["experts.cpp", "experts.cu", "experts_cb0.cu", "experts_cb1.cu", "experts_cb2.cu"]],
                    extra_cflags=["-O3"],
                    extra_cuda_cflags=["-O3", "-lineinfo", f"-DTF_MMA_MINB={mb}",
                                       "-gencode=arch=compute_121,code=sm_121"],
                    build_directory=bd, verbose=False)
    print(f"BUILD_OK {tag} MINB={mb} {time.time()-t0:.1f}s", flush=True)

dev = "cuda"
R, SLOTS = 2048, 7
P = R * SLOTS
D = 5120            # gate/up 的 K = hidden
N = 1152            # TP2 本地 intermediate
E, MAXM = 385, 2048
MAXU = min(P, E)
K2 = 6
TW = 4 * K2
assert D % 64 == 0

WORDS = (D // 16) * (N // 16) * TW
STRIDE_B = WORDS * 4
buf = torch.randint(-2**31, 2**31 - 1, (2 * E * WORDS,), dtype=torch.int32, device=dev)
gate_p = (buf.data_ptr() + torch.arange(E, dtype=torch.int64, device=dev) * STRIDE_B).contiguous()
up_p = (buf.data_ptr() + (E + torch.arange(E, dtype=torch.int64, device=dev)) * STRIDE_B).contiguous()
gk2 = torch.full((E,), K2, dtype=torch.int32, device=dev)
xg = torch.randn(P, D, dtype=torch.float16, device=dev)
xu = torch.randn(P, D, dtype=torch.float16, device=dev)
Z = torch.zeros(2 * P * N, dtype=torch.float32, device=dev)   # 双投影 gate+up
counts = torch.zeros(E, dtype=torch.int32, device=dev)
uids = torch.zeros(MAXU, dtype=torch.int32, device=dev)
ucount = torch.zeros(1, dtype=torch.int32, device=dev)
members = torch.full((MAXU, MAXM), -1, dtype=torch.int32, device=dev)


def list_len(p, maxu, gran):
    return (p + maxu * (gran - 1)) // gran


NGU = list_len(P, MAXU, 64)
work_gu = torch.zeros(NGU * 2, dtype=torch.int32, device=dev)
m0 = mods["base"]
torch.manual_seed(1234)
pick6 = torch.argsort(torch.rand(R, 384, device=dev), dim=1)[:, :6].to(torch.int32)
pick = torch.full((R, SLOTS), 384, dtype=torch.int32, device=dev)
pick[:, :6] = pick6
pick = pick.contiguous()
m0.group_count(pick, counts, R, SLOTS, E)
m0.group_place(pick, counts, uids, ucount, members, R, SLOTS, E)
m0.m2_work(counts, uids, ucount, work_gu, 64)
torch.cuda.synchronize()
print(f"gate/up 形状 K={D} N={N} P={P} ucount={int(ucount.item())} "
      f"权重 {2*E*STRIDE_B/2**30:.3f} GiB 显存 {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)


KBOF = {"base": 4, "new": 2}


def run(tag, kb=None):
    # 真实签名: mats,K,N,P,SK,slots,cb,warps,lo,hi,fold —— 无 kb 参数。
    # KB 走 TF_MMA_KB 运行时覆盖（patch-mma-kb.py），SK=4/warps=4 恒定 ⇒ 只动 KB，干净单因素。
    os.environ["TF_MMA_KB"] = str(kb if kb is not None else KBOF[tag])
    mods[tag].grouped_mma(xg, xu, gate_p, up_p, gk2, gk2, uids, ucount, members, Z,
                         2, D, N, P, 4, SLOTS, 2, 4, K2, K2, 1, work_gu)


if args.only:
    for _ in range(6):
        run(args.only)
    torch.cuda.synchronize()
    print(f"ONLY {args.only} done", flush=True)
    raise SystemExit(0)

# 非空 + KB 数值一致性（KB 只改分块，不应改数值语义；若不一致须标注）
Z.zero_(); run("base"); torch.cuda.synchronize(); z_base = Z.clone()
assert z_base.abs().max().item() > 0, "基线输出全零"
print(f"非空校验 max|Z|={z_base.abs().max().item():.4f}", flush=True)
eq = {}
for tag in ("new",):
    Z.zero_(); run(tag); torch.cuda.synchronize()
    eq[tag] = bool(torch.equal(z_base, Z))
    print(f"  vs 基线 位一致 {tag}: {eq[tag]}  "
          f"maxdiff={(z_base-Z).abs().max().item():.3e}", flush=True)


def timed(tag, reps=7):
    run(tag); torch.cuda.synchronize()
    time.sleep(0.25)          # 臂间冷却：gate/up 连续高负载会触发功耗墙，极差可达 23-49%
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); a = time.perf_counter()
        run(tag)
        torch.cuda.synchronize(); ts.append((time.perf_counter() - a) * 1e3)
        time.sleep(0.05)
    return statistics.median(ts)


ARMS = ["base", "new"]
seq = ARMS * (args.reps // 2)
acc = {t: [] for t in ARMS}
for tag in seq:
    v = timed(tag)
    acc[tag].append(v)
    print(f"  {tag}  {v:.4f} ms", flush=True)

res = {"bit_equal_vs_base": eq, "arms": {}, "rounds": []}
for k, vs in acc.items():
    res["arms"][k] = {"median": statistics.median(vs), "min": min(vs), "max": max(vs),
                      "spread_pct": 100 * (max(vs) - min(vs)) / min(vs)}

b = res["arms"]["base"]["median"]; n = res["arms"]["new"]["median"]
res["delta_pct"] = 100 * (n / b - 1)
res["paired"] = [100*(y/x-1) for x, y in zip(acc["base"], acc["new"])]
res["paired_same_sign"] = all(d < 0 for d in res["paired"]) or all(d > 0 for d in res["paired"])
print(f"\n=== S=2+KB=2+MINB=3 vs 生产同款 ===")
print(f"base(S3,KB4,MB1) {b:.4f} ms   new(S2,KB2,MB3) {n:.4f} ms   delta {res['delta_pct']:+.2f}%")
print(f"配对差同向: {res['paired_same_sign']}  逐对: {[round(d,2) for d in res['paired']]}")
if args.out:
    pathlib_out = args.out
    with open(pathlib_out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"WROTE {pathlib_out}")
