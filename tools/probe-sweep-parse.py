#!/usr/bin/env python3
"""E-SELF-09：S×MINB 扫描的机制读数横表。"""
import csv, os, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--arms", default="0-1,3-1,3-2,6-1")
ap.add_argument("--roof", type=float, default=246.2)
args = ap.parse_args()
os.chdir(os.path.expanduser("~/ptxwork"))


def load(tag):
    lines = open(f"sw-{tag}.csv").read().splitlines()
    s = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    d = {}
    for r in csv.DictReader(lines[s:]):
        try:
            d[r["Metric Name"]] = float(r["Metric Value"].replace(",", ""))
        except (ValueError, KeyError):
            pass
    return d


arms = args.arms.split(",")
D = {a: load(a) for a in arms}
for a in arms:
    assert D[a], f"{a} 解析为空"

rows = [
    ("时长 ms", lambda d: d["gpu__time_duration.sum"] / 1e6, "{:.4f}"),
    ("REG/thread", lambda d: d.get("launch__registers_per_thread", 0), "{:.0f}"),
    ("dynamic shared B", lambda d: d.get("launch__shared_mem_per_block_dynamic", 0), "{:.0f}"),
    ("occ_limit_regs", lambda d: d.get("launch__occupancy_limit_registers", 0), "{:.0f}"),
    ("occ_limit_shared", lambda d: d.get("launch__occupancy_limit_shared_mem", 0), "{:.0f}"),
    ("达成占用率 %", lambda d: d.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0), "{:.2f}"),
    ("DRAM miss GiB", lambda d: d["lts__t_sectors_op_read_lookup_miss.sum"] * 32 / 2**30, "{:.3f}"),
    ("达成带宽 GB/s", lambda d: d["lts__t_sectors_op_read_lookup_miss.sum"] * 32 / (d["gpu__time_duration.sum"] / 1e9) / 1e9, "{:.1f}"),
    ("占屋顶 %", lambda d: 100 * d["lts__t_sectors_op_read_lookup_miss.sum"] * 32 / (d["gpu__time_duration.sum"] / 1e9) / 1e9 / args.roof, "{:.1f}"),
    ("L2 吞吐 %", lambda d: d.get("lts__throughput.avg.pct_of_peak_sustained_elapsed", 0), "{:.2f}"),
    ("L1TEX 吞吐 %", lambda d: d.get("l1tex__throughput.avg.pct_of_peak_sustained_elapsed", 0), "{:.2f}"),
    ("SM 吞吐 %", lambda d: d.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0), "{:.2f}"),
    ("Tensor %", lambda d: d.get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", 0), "{:.2f}"),
    ("总指令", lambda d: d.get("smsp__inst_executed.sum", 0), "{:,.0f}"),
]
print(f"{'量':<20}" + "".join(f"{a:>14}" for a in arms))
for name, fn, fmt in rows:
    print(f"{name:<20}" + "".join(f"{fmt.format(fn(D[a])):>14}" for a in arms))

print(f"\n{'stall 占比 %':<20}" + "".join(f"{a:>14}" for a in arms))
S = [k for k in D[arms[0]] if "issue_stalled" in k and k.endswith(".pct")]
tot = {a: sum(D[a][k] for k in S) for a in arms}
for k in sorted(S, key=lambda k: -D[arms[0]][k]):
    r = k.replace("smsp__average_warp_latency_issue_stalled_", "").replace(".pct", "")
    print(f"{r:<20}" + "".join(f"{100*D[a].get(k,0)/tot[a]:>13.1f}%" for a in arms))