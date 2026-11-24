#!/usr/bin/env python3
"""全局 Amdahl 重算：从 steady trace 逐内核归并，分类、算占比、算空隙结构。

口径纪律：
  - timeline.json 的 kernel 记录是 [start_ns, end_ns, stream, name, gx, gy, gz]
    （stream 在 index 2，name 在 index 3 —— 早期脚本曾把 name 当 stream，已纠正）
  - 分母同时给「请求窗口」与「union busy」两个，避免把空隙混进内核占比
  - 只取稳定请求（req2/req3），req1 是预热
"""
import json, collections, argparse, re

ap = argparse.ArgumentParser()
ap.add_argument("--trace", default="runs/forge-20261123-steady-trace/timeline.json")
ap.add_argument("--req", default="req2")
ap.add_argument("--top", type=int, default=40)
ap.add_argument("--gaps", type=int, default=12)
ap.add_argument("--name", default="", help="只打印匹配该正则的内核明细")
a = ap.parse_args()

SL = {"req1": (24, 6104), "req2": (6104, 12184), "req3": (12184, 18264)}
K = json.load(open(a.trace))["kernels"]
lo, hi = SL[a.req]
seg = K[lo:hi]

t0 = min(x[0] for x in seg); t1 = max(x[1] for x in seg)
window = t1 - t0
iv = sorted((x[0], x[1]) for x in seg)
busy = 0; cs, ce = iv[0]; merged = []
for s, e in iv[1:]:
    if s > ce:
        merged.append((cs, ce)); busy += ce - cs; cs, ce = s, e
    else:
        ce = max(ce, e)
merged.append((cs, ce)); busy += ce - cs


def short(n):
    if n.startswith("ncclDevKernel"):
        return "nccl:" + n.split("_")[1]
    if "grouped_mma_kernel" in n: return "exl3:grouped_mma (gate/up)"
    if "grouped_rows_kernel" in n: return "exl3:grouped_rows (down)"
    if "gateup_epilogue" in n: return "exl3:gateup_epilogue"
    if "down_combine" in n: return "exl3:down_combine"
    if "group_count" in n: return "exl3:group_count"
    if "group_place" in n: return "exl3:group_place"
    if "work_list" in n: return "exl3:work_list"
    if "rot_in_kernel<__nv_bfloat16>" in n: return "exl3:rot_in<bf16>"
    if "rot_in_kernel(const void" in n: return "exl3:rot_in<void>"
    if "rot_many" in n: return "exl3:rot_many"
    m = re.search(r"unpack_kernel<(int)(\d+), \(int\)(\d+)>", n)
    if m: return f"exl3:unpack<{m.group(2)},{m.group(3)}>"
    m = re.search(r"glinear_kernel<\(int\)(\d+), \(int\)(\d+), \(int\)(\d+)>", n)
    if m: return f"exl3:glinear<{m.group(1)},{m.group(2)},{m.group(3)}>"
    if "simt_sgemm" in n:
        m = re.search(r"simt_sgemm_(\d+x\d+_\d+x\d+)", n)
        return "cutlass:simt_sgemm_" + (m.group(1) if m else "?")
    if "nvjet_" in n:
        m = re.search(r"nvjet_sm121_tst_mma_(\d+x\d+x\d+)", n)
        return "nvjet:" + (m.group(1) if m else "?")
    if "splitKreduce" in n: return "cublasLt:splitKreduce"
    return n


agg = collections.defaultdict(lambda: [0, 0])
grids = collections.defaultdict(collections.Counter)
for s, e, st, n, gx, gy, gz in seg:
    k = short(n)
    agg[k][0] += e - s; agg[k][1] += 1
    grids[k][(gx, gy, gz)] += 1

CAT = [
    ("专家 exl3 核心", lambda k: k.startswith("exl3:grouped")),
    ("专家 exl3 周边", lambda k: k.startswith("exl3:") and not k.startswith("exl3:grouped")),
    ("集合通信", lambda k: k.startswith("nccl:")),
    ("稠密 GEMM", lambda k: k.startswith(("nvjet:", "cutlass:", "cublasLt:")) or k == "_gemm"),
    ("注意力/HC", lambda k: k.startswith(("_sparse_attn", "_hc_", "_rope", "_kv_norm", "_index_score", "_topk", "_collapse"))),
    ("路由/索引", lambda k: k.startswith(("_route", "tf_ds_scatter", "tf_gather", "tf_ds_topk", "tf_ds_copy_rows"))),
    ("norm/cast/量化", lambda k: k.startswith(("_rmsnorm", "tf_ds_rms", "tf_bf16", "tf_ds_f16", "tf_ds_bf16", "tf_ds_compress", "_fp4", "tf_ds_add2", "tf_ds_rope"))),
]


def cat_of(k):
    for nm, f in CAT:
        if f(k): return nm
    return "其它"


print(f"== {a.req} ==  窗口 {window/1e6:.3f} ms   union busy {busy/1e6:.3f} ms   空隙 {(window-busy)/1e6:.3f} ms ({100*(window-busy)/window:.2f}%)")
print(f"内核数 {len(seg)}   不同名 {len(agg)}   stream {sorted(set(x[2] for x in seg))}")
print()
print(f"{'内核':<34}{'总ms':>10}{'次':>6}{'均ms':>9}{'占窗口%':>9}{'占busy%':>9}")
rows = sorted(agg.items(), key=lambda kv: -kv[1][0])
for k, (d, c) in rows[:a.top]:
    print(f"{k:<34}{d/1e6:>10.2f}{c:>6}{d/c/1e6:>9.4f}{100*d/window:>9.2f}{100*d/busy:>9.2f}")
tail = sum(d for k, (d, c) in rows[a.top:])
print(f"{'(其余 '+str(len(rows)-a.top)+' 名合计)':<34}{tail/1e6:>10.2f}{'':>6}{'':>9}{100*tail/window:>9.2f}{100*tail/busy:>9.2f}")

print()
print("== 分类汇总 ==")
cg = collections.defaultdict(int)
for k, (d, c) in agg.items(): cg[cat_of(k)] += d
for nm, d in sorted(cg.items(), key=lambda kv: -kv[1]):
    print(f"{nm:<20}{d/1e6:>10.2f} ms{100*d/window:>8.2f}% 窗口{100*d/busy:>8.2f}% busy")
print(f"{'空隙(no-kernel)':<20}{(window-busy)/1e6:>10.2f} ms{100*(window-busy)/window:>8.2f}% 窗口")

print()
print(f"== 最大 {a.gaps} 个空隙（前后各一个内核，判归属）==")
gp = []
for i in range(1, len(merged)):
    g = merged[i][0] - merged[i-1][1]
    if g > 0: gp.append((g, merged[i-1][1], merged[i][0]))
gp.sort(reverse=True)
byn = {}
for s, e, st, n, gx, gy, gz in seg: byn.setdefault(s, short(n))
prev_end = {}
for s, e, st, n, gx, gy, gz in seg: prev_end[e] = short(n)
for g, ge, gs in gp[:a.gaps]:
    print(f"{g/1e6:>9.3f} ms  @+{(ge-t0)/1e6:>9.1f}ms  后接 {byn.get(gs,'?')}")
print(f"空隙总和 {sum(x[0] for x in gp)/1e6:.3f} ms，其中 >1ms 的 {sum(x[0] for x in gp if x[0]>1e6)/1e6:.3f} ms（{sum(1 for x in gp if x[0]>1e6)} 个）")

if a.name:
    print()
    print(f"== 明细：{a.name} ==")
    for k, (d, c) in rows:
        if re.search(a.name, k):
            print(f"{k}  总 {d/1e6:.2f} ms  {c} 次  均 {d/c/1e6:.4f} ms")
            for g, n in grids[k].most_common(8): print(f"    grid {g} × {n}")