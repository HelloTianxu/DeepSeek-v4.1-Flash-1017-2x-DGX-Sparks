#!/usr/bin/env python3
"""E-ROWS-01：MMA_ROWS 64→32 源码级适配后的判决实验。

三模块（串行编译，MAX_JOBS=2）：
  ref : exl3src       原件，未打任何 patch          —— 正确性金标准
  r64 : exl3src-rows  已参数化，-DTF_MMA_ROWS=64    —— 回归臂，须与 ref 位一致
  r32 : exl3src-rows  同一树，  -DTF_MMA_ROWS=32 -DTF_MMA_MINB=3

两道位一致闸：
  ref vs r64 ⇒ patch 在 R=64 下逐字等价（参数化没改语义）
  r64 vs r32 ⇒ R=32 的行映射/work/epilogue 适配正确
任一不过 ⇒ 立即停，不做性能结论。

work 按各自 rows 生成（m2_work 本就带 rows 参数，这是第五处耦合的正解）。
KB 由 chain=(K/16)/(SK*warps)=20 自动推为 4，两臂相同。

用法：python3 probe-rows.py [--reps 12] [--out rows.json] [--only r64|r32]
"""
import os, json, argparse, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=12)
ap.add_argument("--out", default="")
ap.add_argument("--only", default="")
args = ap.parse_args()

BUILD_ROOT = os.path.expanduser("~/ptxwork")
os.environ.setdefault("MAX_JOBS", "2")
import torch
from torch.utils.cpp_extension import load

SRCS = ["experts.cpp", "experts.cu", "experts_cb0.cu", "experts_cb1.cu", "experts_cb2.cu"]
# tag: (源码树, build 目录, 额外宏)
CFG = {
    "ref": ("exl3src",      "probe-build-rows-ref", []),
    "r64": ("exl3src-rows", "probe-build-rows-r64", ["-DTF_MMA_ROWS=64", "-DTF_MMA_MINB=1"]),
    "r32": ("exl3src-rows", "probe-build-rows-r32", ["-DTF_MMA_ROWS=32", "-DTF_MMA_MINB=3"]),
}
mods = {}
for tag, (sd, bdn, macros) in CFG.items():
    src = os.path.join(BUILD_ROOT, sd)
    bd = os.path.join(BUILD_ROOT, bdn)
    os.makedirs(bd, exist_ok=True)
    t0 = time.time()
    mods[tag] = load(name=f"tf_exl3_{bdn.replace(chr(45), chr(95))}",
                     sources=[os.path.join(src, n) for n in SRCS],
                     extra_cflags=["-O3"],
                     extra_cuda_cflags=["-O3", "-lineinfo", *macros,
                                        "-gencode=arch=compute_121,code=sm_121"],
                     build_directory=bd, verbose=False)
    print(f"BUILD_OK {tag:4s} {sd:14s} {' '.join(macros):32s} {time.time()-t0:.1f}s", flush=True)

dev = "cuda"
R, SLOTS = 2048, 7
P = R * SLOTS
D, N = 5120, 1152
E, MAXM = 385, 2048
MAXU = min(P, E)
K2 = 6
TW = 4 * K2
WORDS = (D // 16) * (N // 16) * TW
STRIDE_B = WORDS * 4
buf = torch.randint(-2**31, 2**31 - 1, (2 * E * WORDS,), dtype=torch.int32, device=dev)
gate_p = (buf.data_ptr() + torch.arange(E, dtype=torch.int64, device=dev) * STRIDE_B).contiguous()
up_p = (buf.data_ptr() + (E + torch.arange(E, dtype=torch.int64, device=dev)) * STRIDE_B).contiguous()
gk2 = torch.full((E,), K2, dtype=torch.int32, device=dev)
xg = torch.randn(P, D, dtype=torch.float16, device=dev)
xu = torch.randn(P, D, dtype=torch.float16, device=dev)
Z = torch.zeros(2 * P * N, dtype=torch.float32, device=dev)
counts = torch.zeros(E, dtype=torch.int32, device=dev)
uids = torch.zeros(MAXU, dtype=torch.int32, device=dev)
ucount = torch.zeros(1, dtype=torch.int32, device=dev)
members = torch.full((MAXU, MAXM), -1, dtype=torch.int32, device=dev)

torch.manual_seed(1234)
pick6 = torch.argsort(torch.rand(R, 384, device=dev), dim=1)[:, :6].to(torch.int32)
pick = torch.full((R, SLOTS), 384, dtype=torch.int32, device=dev)
pick[:, :6] = pick6
pick = pick.contiguous()
m0 = mods["ref"]
m0.group_count(pick, counts, R, SLOTS, E)
m0.group_place(pick, counts, uids, ucount, members, R, SLOTS, E)


def list_len(p, maxu, gran):
    return (p + maxu * (gran - 1)) // gran


WORK = {}
for rows in (64, 32):
    n = list_len(P, MAXU, rows)
    w = torch.zeros(n * 2, dtype=torch.int32, device=dev)
    m0.m2_work(counts, uids, ucount, w, rows)
    WORK[rows] = w
torch.cuda.synchronize()
print(f"形状 K={D} N={N} P={P} ucount={int(ucount.item())} "
      f"work对数 64行={WORK[64].numel()//2} 32行={WORK[32].numel()//2} "
      f"权重 {2*E*STRIDE_B/2**30:.3f} GiB 显存 {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)

ROWSOF = {"ref": 64, "r64": 64, "r32": 32}


def run(tag):
    mods[tag].grouped_mma(xg, xu, gate_p, up_p, gk2, gk2, uids, ucount, members, Z,
                          2, D, N, P, 4, SLOTS, 2, 4, K2, K2, 1, WORK[ROWSOF[tag]])


if args.only:
    assert args.only in CFG, f"--only 须是 {list(CFG)}"
    for _ in range(6):
        run(args.only)
    torch.cuda.synchronize()
    print(f"ONLY {args.only} done", flush=True)
    raise SystemExit(0)

# ---- 两道位一致闸 ----
Z.zero_(); run("ref"); torch.cuda.synchronize(); z_ref = Z.clone()
mx = z_ref.abs().max().item()
assert mx > 0, "参考输出全零"
print(f"非空校验 max|Z|={mx:.4f}", flush=True)
eq = {}
for tag in ("r64", "r32"):
    Z.zero_(); run(tag); torch.cuda.synchronize()
    eq[tag] = bool(torch.equal(z_ref, Z))
    print(f"  闸{'1 回归' if tag=='r64' else '2 适配'} ref vs {tag}: 位一致={eq[tag]} "
          f"maxdiff={(z_ref-Z).abs().max().item():.3e}", flush=True)
if not all(eq.values()):
    print("GATE_FAIL 位一致不过，停止性能测量", flush=True)
    if args.out:
        json.dump({"bit_equal_vs_ref": eq, "gate": "FAIL"}, open(args.out, "w"), indent=1)
    raise SystemExit(2)


def timed(tag, reps=7):
    run(tag); torch.cuda.synchronize()
    time.sleep(0.25)                      # 臂间冷却（手册 §7.22）
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); a = time.perf_counter()
        run(tag)
        torch.cuda.synchronize(); ts.append((time.perf_counter() - a) * 1e3)
        time.sleep(0.05)
    return statistics.median(ts)


ARMS = ["r64", "r32"]
acc = {t: [] for t in ARMS}
for tag in ARMS * (args.reps // 2):
    v = timed(tag)
    acc[tag].append(v)
    print(f"  {tag}  {v:.4f} ms", flush=True)

res = {"bit_equal_vs_ref": eq, "gate": "PASS", "arms": {}}
for k, vs in acc.items():
    res["arms"][k] = {"median": statistics.median(vs), "min": min(vs), "max": max(vs),
                      "spread_pct": 100 * (max(vs) - min(vs)) / min(vs)}
b = res["arms"]["r64"]["median"]; n = res["arms"]["r32"]["median"]
res["delta_pct"] = 100 * (n / b - 1)
res["paired"] = [100 * (y / x - 1) for x, y in zip(acc["r64"], acc["r32"])]
res["paired_same_sign"] = all(d < 0 for d in res["paired"]) or all(d > 0 for d in res["paired"])
print(f"\n=== E-ROWS-01 判决：MMA_ROWS 64(2 blocks) vs 32(3 blocks) ===")
print(f"r64 {b:.4f} ms   r32 {n:.4f} ms   delta {res['delta_pct']:+.2f}%")
print(f"配对差同向: {res['paired_same_sign']}  逐对: {[round(d,2) for d in res['paired']]}")
print(f"模型预期（带宽 150.49→175.2 GB/s）: {-100*(1-1/1.1642):+.2f}%")
if args.out:
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"WROTE {args.out}")