#!/usr/bin/env python3
"""E-OCC-03：三点 occupancy 扫描，钉死 T = A/warp + B 的不可掩盖基座 B。

E-OCC-02 两点（blocks 2/1，指令不变位一致）给出 B≈8.855ms，接近 DRAM 硬下界 8.118ms。
两点拟合必然完美，须第三点验证。KB=2+MINB=3 模块 shared 23808 / REG 80，
可用 dyn pad 做出 blocks 3/2/1 三档，**三档之间指令完全相同**（同模块同 KB）。
若三点拟合的 B 与 E-OCC-02 的 B 一致 ⇒ B 是真实的物理基座（DRAM 传输），归因闭合。

  pad=0     ⇒ shared 23808 ⇒ blocks min(3,4)=3
  pad=12000 ⇒ shared 35808 ⇒ blocks min(3,2)=2
  pad=45000 ⇒ shared 68808 ⇒ blocks min(3,1)=1
用法：python3 probe-dyn3.py [--reps 9] [--out dyn3.json] [--only b3|b2|b1]
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
TREES = {"dyn": ("exl3src-dyn3", "probe-build-dyn3", 3)}
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
m0 = mods["dyn"]
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


PADOF = {"b3": 0, "b2": 12000, "b1": 45000}


def run(tag, kb=None):
    # 真实签名: mats,K,N,P,SK,slots,cb,warps,lo,hi,fold —— 无 kb 参数。
    # KB 走 TF_MMA_KB 运行时覆盖（patch-mma-kb.py），SK=4/warps=4 恒定 ⇒ 只动 KB，干净单因素。
    os.environ["TF_MMA_KB"] = "2"
    os.environ["TF_MMA_DYN_PAD"] = str(PADOF[tag])
    mods["dyn"].grouped_mma(xg, xu, gate_p, up_p, gk2, gk2, uids, ucount, members, Z,
                         2, D, N, P, 4, SLOTS, 2, 4, K2, K2, 1, work_gu)


if args.only:
    for _ in range(6):
        run(args.only)
    torch.cuda.synchronize()
    print(f"ONLY {args.only} done", flush=True)
    raise SystemExit(0)

# 非空 + KB 数值一致性（KB 只改分块，不应改数值语义；若不一致须标注）
Z.zero_(); run("b3"); torch.cuda.synchronize(); z_base = Z.clone()
assert z_base.abs().max().item() > 0, "基线输出全零"
print(f"非空校验 max|Z|={z_base.abs().max().item():.4f}", flush=True)
eq = {}
for tag in ("b2", "b1"):
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


ARMS = ["b3", "b2", "b1"]
seq = ARMS * (args.reps // 3)
acc = {t: [] for t in ARMS}
for tag in seq:
    v = timed(tag)
    acc[tag].append(v)
    print(f"  {tag}  {v:.4f} ms", flush=True)

res = {"bit_equal_vs_base": eq, "arms": {}, "rounds": []}
for k, vs in acc.items():
    res["arms"][k] = {"median": statistics.median(vs), "min": min(vs), "max": max(vs),
                      "spread_pct": 100 * (max(vs) - min(vs)) / min(vs)}

b = res["arms"]["b3"]["median"]; n = res["arms"]["b1"]["median"]
res["delta_pct"] = 100 * (n / b - 1)
res["paired"] = [100*(y/x-1) for x, y in zip(acc["b3"], acc["b1"])]
res["paired_same_sign"] = all(d < 0 for d in res["paired"]) or all(d > 0 for d in res["paired"])
print(f"\n=== E-OCC-03 三点 occupancy 扫描（KB=2 固定，指令三档相同）===")
print(f"blocks3 {b:.4f} ms   blocks1 {n:.4f} ms   delta {res['delta_pct']:+.2f}%")
print(f"配对差同向: {res['paired_same_sign']}  逐对: {[round(d,2) for d in res['paired']]}")
for k, v in res["arms"].items():
    print(f"  {k}: median {v['median']:.4f} ms  spread {v['spread_pct']:.2f}%")
if args.out:
    pathlib_out = args.out
    with open(pathlib_out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"WROTE {pathlib_out}")
