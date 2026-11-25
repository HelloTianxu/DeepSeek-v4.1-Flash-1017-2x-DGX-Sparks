#!/usr/bin/env python3
"""E-OCC-01 实验B：gate/up 的 KB × MINB 2×2 因子设计，判决「并发不足」是否为主因。

编译期预筛（生产同款 range HI=10，cuobjdump -res-usage 实读）：
  KB=4, MINB=1: REG 124, spill 0,  shared 45312  ⇒ blocks = min(reg 2, sh 2) = 2  ← 生产
  KB=4, MINB=3: REG  80, spill 24, shared 45312  ⇒ shared 仍钉 2，白付 spill
  KB=2, MINB=1: REG 112, spill 0,  shared 24832  ⇒ blocks = min(reg 2, sh 4) = 2
  KB=2, MINB=3: REG  79, spill 16, shared 24832  ⇒ blocks = min(reg 3, sh 4) = 3  ← 唯一可提并发
只有 KB=2+MINB=3 能真正把 blocks 2→3（占用 16/48=33.3% → 24/48=50%）。
2×2 可分离「KB 减半的权重重读代价」与「并发 1.5× 的收益」。

MINB 是编译期宏 ⇒ 两个独立模块（同进程加载，共享同一份权重 tensor）⇒ 真正同场。
用法：python3 probe-occ-2x2.py [--reps 6] [--out occ.json]
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
for mb in (1, 3):
    src = os.path.join(BUILD_ROOT, f"exl3src-mb{mb}")
    bd = os.path.join(BUILD_ROOT, f"probe-build-mb{mb}")
    os.makedirs(bd, exist_ok=True)
    t0 = time.time()
    mods[mb] = load(name=f"tf_exl3_mb{mb}",
                    sources=[os.path.join(src, n) for n in
                             ["experts.cpp", "experts.cu", "experts_cb0.cu", "experts_cb1.cu", "experts_cb2.cu"]],
                    extra_cflags=["-O3"],
                    extra_cuda_cflags=["-O3", "-lineinfo", f"-DTF_MMA_MINB={mb}",
                                       "-gencode=arch=compute_121,code=sm_121"],
                    build_directory=bd, verbose=False)
    print(f"BUILD_OK MINB={mb} {time.time()-t0:.1f}s", flush=True)

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
m0 = mods[1]
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


def run(mb, kb):
    # 真实签名: mats,K,N,P,SK,slots,cb,warps,lo,hi,fold —— 无 kb 参数。
    # KB 走 TF_MMA_KB 运行时覆盖（patch-mma-kb.py），SK=4/warps=4 恒定 ⇒ 只动 KB，干净单因素。
    os.environ["TF_MMA_KB"] = str(kb)
    mods[mb].grouped_mma(xg, xu, gate_p, up_p, gk2, gk2, uids, ucount, members, Z,
                         2, D, N, P, 4, SLOTS, 2, 4, K2, K2, 1, work_gu)


if args.only:
    mb, kb = (int(x) for x in args.only.split(","))
    for _ in range(6):
        run(mb, kb)
    torch.cuda.synchronize()
    print(f"ONLY mb{mb}_kb{kb} done", flush=True)
    raise SystemExit(0)

# 非空 + KB 数值一致性（KB 只改分块，不应改数值语义；若不一致须标注）
Z.zero_(); run(1, 4); torch.cuda.synchronize(); z_base = Z.clone()
assert z_base.abs().max().item() > 0, "基线输出全零"
print(f"非空校验 max|Z|={z_base.abs().max().item():.4f}", flush=True)
eq = {}
for mb, kb in [(1, 2), (3, 4), (3, 2)]:
    Z.zero_(); run(mb, kb); torch.cuda.synchronize()
    eq[f"mb{mb}_kb{kb}"] = bool(torch.equal(z_base, Z))
    print(f"  vs 基线 位一致 mb{mb}_kb{kb}: {eq[f'mb{mb}_kb{kb}']}  "
          f"maxdiff={(z_base-Z).abs().max().item():.3e}", flush=True)


def timed(mb, kb, reps=7):
    run(mb, kb); torch.cuda.synchronize()
    time.sleep(0.25)          # 臂间冷却：gate/up 连续高负载会触发功耗墙，极差可达 23-49%
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); a = time.perf_counter()
        run(mb, kb)
        torch.cuda.synchronize(); ts.append((time.perf_counter() - a) * 1e3)
        time.sleep(0.05)
    return statistics.median(ts)


ARMS = [(1, 4), (1, 2), (3, 4), (3, 2)]
seq = ARMS * (args.reps // 2)
acc = {f"mb{mb}_kb{kb}": [] for mb, kb in ARMS}
for mb, kb in seq:
    v = timed(mb, kb)
    acc[f"mb{mb}_kb{kb}"].append(v)
    print(f"  mb{mb}_kb{kb}  {v:.4f} ms", flush=True)

res = {"bit_equal_vs_base": eq, "arms": {}, "rounds": []}
for k, vs in acc.items():
    res["arms"][k] = {"median": statistics.median(vs), "min": min(vs), "max": max(vs),
                      "spread_pct": 100 * (max(vs) - min(vs)) / min(vs)}
base = res["arms"]["mb1_kb4"]["median"]
print("\n=== 2x2 结果（相对生产 mb1_kb4）===")
for k in ["mb1_kb4", "mb1_kb2", "mb3_kb4", "mb3_kb2"]:
    a = res["arms"][k]
    print(f"  {k}: median {a['median']:.4f} ms  {100*(a['median']-base)/base:+.3f}%  极差 {a['spread_pct']:.2f}%")
b = res["arms"]
kb_cost = 100 * (b["mb1_kb2"]["median"] - b["mb1_kb4"]["median"]) / b["mb1_kb4"]["median"]
minb_at_kb2 = 100 * (b["mb3_kb2"]["median"] - b["mb1_kb2"]["median"]) / b["mb1_kb2"]["median"]
print(f"\nKB 4->2 净代价（MINB=1 固定）: {kb_cost:+.3f}%")
print(f"MINB 1->3 净效应（KB=2 固定，blocks 2->3）: {minb_at_kb2:+.3f}%")
res["kb_cost_pct"] = kb_cost
res["minb_effect_at_kb2_pct"] = minb_at_kb2
if args.out:
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"WROTE {args.out}", flush=True)