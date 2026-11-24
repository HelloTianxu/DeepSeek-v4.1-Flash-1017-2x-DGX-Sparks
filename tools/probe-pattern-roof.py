#!/usr/bin/env python3
"""E-SELF-09c：访存形状屋顶标定驱动。三内核读相同总字节，只差形状。

stream  = 顺序流式（本进程屋顶复标）
pattern = mma2 真实跨距形状，无解码无 MMA
slab    = 同字节同程序数，但每程序连续 ⇒ 布局改私有连续的上限
"""
import os, json, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2304)
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--out", default="")
ap.add_argument("--only", default="")
ap.add_argument("--ncu-warm", action="store_true")
args = ap.parse_args()

import torch
from torch.utils.cpp_extension import load

SRC = os.path.expanduser("~/ptxwork")
OUT = os.path.expanduser("~/ptxwork/probe-build-roof")
os.makedirs(OUT, exist_ok=True)
r = load(name="tf_pattern_roof", sources=[os.path.join(SRC, "probe-pattern-roof.cu")],
         extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121,code=sm_121"],
         extra_cflags=["-O3"], build_directory=OUT, verbose=False)

dev = "cuda"
R, SLOTS = 2048, 7
P = R * SLOTS
D = 5120
N = args.n
E, MAXM = 385, 2048
MAXU = min(P, E)
K2 = 6
TW = 4 * K2                 # 24 words
M2T = 16                    # M2_TILES
KT = D // 16                # 320
NTILES = N // 16            # 144
NGU = (P + MAXU * 63) // 64
WORDS = KT * NTILES * TW
STRIDE_B = WORDS * 4

buf = torch.randint(-2**31, 2**31 - 1, (E * WORDS,), dtype=torch.int32, device=dev)
ptrs = (buf.data_ptr() + torch.arange(E, dtype=torch.int64, device=dev) * STRIDE_B).contiguous()
counts = torch.zeros(E, dtype=torch.int32, device=dev)
uids = torch.zeros(MAXU, dtype=torch.int32, device=dev)
ucount = torch.zeros(1, dtype=torch.int32, device=dev)
members = torch.full((MAXU, MAXM), -1, dtype=torch.int32, device=dev)
work = torch.zeros(NGU * 2, dtype=torch.int32, device=dev)

# 路由与 probe-m2-sweep 完全一致（同 seed 同构造），保证早退/活跃程序集合相同
big = torch.zeros(NGU * 2, dtype=torch.int32, device=dev)
torch.manual_seed(1234)
pick6 = torch.argsort(torch.rand(R, 384, device=dev), dim=1)[:, :6].to(torch.int32)
pick = torch.full((R, SLOTS), 384, dtype=torch.int32, device=dev)
pick[:, :6] = pick6
pick = pick.contiguous()

# 复用已编译的 sweep 模块构造工作列表（同一套 group_count/place/m2_work）
mm = load(name="tf_exl3_sweep",
          sources=[os.path.join(os.path.expanduser("~/ptxwork/exl3src-sweep"), n) for n in
                   ["experts.cpp", "experts.cu", "experts_cb0.cu", "experts_cb1.cu", "experts_cb2.cu"]],
          extra_cflags=["-O3"],
          extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121,code=sm_121"],
          build_directory=os.path.expanduser("~/ptxwork/probe-build-sweep"), verbose=False)
mm.group_count(pick, counts, R, SLOTS, E)
mm.group_place(pick, counts, uids, ucount, members, R, SLOTS, E)
mm.m2_work(counts, uids, ucount, work, 64)
torch.cuda.synchronize()

nu = int(ucount.item())
live = counts[uids[:nu].to(torch.int64)].to(torch.int64)
live_gu = int(((live + 63) // 64).sum())
NY, NZ = N // 256, NGU * 2
CH = M2T * TW // 4                      # 96 uint4
prog_bytes = KT * CH * 16               # 每程序字节
# 活跃程序数：每个活跃 (expert, member-block) 有 NY 个列块，两投影
n_active = live_gu * NY * 2
tot_bytes = n_active * prog_bytes
print(f"N={N} KT={KT} NTILES={NTILES} TW={TW}  活跃组={live_gu}/{NGU}")
print(f"每程序 {prog_bytes/1024:.0f} KiB（{KT} 块 × {CH*16} B，跨距 {NTILES*TW*4} B）")
print(f"活跃程序 {n_active}  总读字节 {tot_bytes/2**30:.3f} GiB")
print(f"缓冲 {buf.numel()*4/2**30:.3f} GiB  显存 {torch.cuda.memory_allocated()/2**30:.2f} GiB")

out_s = torch.zeros(4096, dtype=torch.int32, device=dev)
out_g = torch.zeros(NY * NZ, dtype=torch.int32, device=dev)


def run_stream():
    r.stream(buf, out_s, 1024)


def run_pattern():
    r.pattern(ptrs, uids, ucount, members, work, out_g, KT, NTILES, TW, M2T, 64, NY, NZ)


def run_slab():
    r.slab(ptrs, uids, ucount, members, work, out_g, KT, NTILES, TW, M2T, 64, NY, NZ)


xact = torch.randn(P, D, dtype=torch.float16, device=dev)


def run_full():
    r.full(ptrs, uids, ucount, members, work, xact, out_g, KT, NTILES, TW, M2T, 64, NY, NZ, D, SLOTS, 4)


FN = {"stream": run_stream, "pattern": run_pattern, "slab": run_slab, "full": run_full}
# stream 读整个缓冲（可能大于活跃程序总字节），单独算它的字节数
# full = 权重 + 激活（激活按 live 行、每列块重复 staging，与内核一致）
act_rows = int(((live + 15) // 16).sum()) * 16
act_bytes = act_rows * NY * 2 * D * 2
BYTES = {"stream": buf.numel() * 4, "pattern": tot_bytes, "slab": tot_bytes,
         "full": tot_bytes + act_bytes}
print(f"full 字节 = 权重 {tot_bytes/2**30:.3f} + 激活 {act_bytes/2**30:.3f} = {(tot_bytes+act_bytes)/2**30:.3f} GiB"
      f"（激活 live 行 {act_rows}，被 {NY} 个列块各 staging 一遍）")


def timeit(name):
    fn = FN[name]
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(args.reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ts.sort()
    return ts[len(ts) // 2], ts[0], ts[-1]


names = [args.only] if args.only else ["stream", "pattern", "slab", "full", "pattern", "full"]
res = {"N": N, "total_GiB": tot_bytes / 2**30, "n_active": n_active, "runs": []}
print(f"\n{'内核':<10}{'中位ms':>10}{'min':>9}{'字节GiB':>10}{'达成GB/s':>11}{'按min':>10}")
for nm in names:
    med, lo, hi = timeit(nm)
    b = BYTES[nm]
    bw = b / (med / 1e3) / 1e9
    bwm = b / (lo / 1e3) / 1e9
    print(f"{nm:<10}{med:>10.4f}{lo:>9.4f}{b/2**30:>10.3f}{bw:>11.1f}{bwm:>10.1f}")
    res["runs"].append({"kernel": nm, "median_ms": med, "min_ms": lo, "bytes": b,
                        "GBps_median": bw, "GBps_min": bwm})

if not args.only:
    pat = [x for x in res["runs"] if x["kernel"] == "pattern"]
    slb = [x for x in res["runs"] if x["kernel"] == "slab"]
    stm = [x for x in res["runs"] if x["kernel"] == "stream"]
    if pat and slb and stm:
        p = max(x["GBps_min"] for x in pat)
        s = max(x["GBps_min"] for x in slb)
        t = max(x["GBps_min"] for x in stm)
        print(f"\n屋顶复标(stream) {t:.1f} GB/s")
        print(f"形状屋顶(pattern) {p:.1f} GB/s = stream 的 {100*p/t:.1f}%")
        print(f"连续布局(slab)    {s:.1f} GB/s = stream 的 {100*s/t:.1f}%，pattern 的 {s/p:.3f}×")
        res["roof"] = {"stream": t, "pattern": p, "slab": s}
        ful = [x for x in res["runs"] if x["kernel"] == "full"]
        if ful:
            fm = min(x["min_ms"] for x in ful)
            fb = max(x["GBps_min"] for x in ful)
            print(f"**完整 staged 形状屋顶(full)** {fb:.1f} GB/s，耗时 {fm:.4f} ms")
            print(f"mma2 完整实测 22.73 ms ⇒ 是 full 屋顶的 {22.73/fm:.3f}× "
                  f"⇒ 递货 plumbing 之外还差 {22.73-fm:+.2f} ms")
            res["roof"]["full_Gbps"] = fb
            res["roof"]["full_ms"] = fm

if args.out:
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("WROTE", args.out)
print("PROBE_OK")