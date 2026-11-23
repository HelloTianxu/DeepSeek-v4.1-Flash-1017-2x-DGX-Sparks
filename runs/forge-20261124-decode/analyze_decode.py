#!/usr/bin/env python3
"""E-DECODE-01：decode 窗口内真实 DRAM 带宽与屋顶占比。

沿用 E-DRAM-01 r2 已校准口径：VidL2 Misses from L1TEX 原始值为 sectors/s，
GB/s = raw × 32 / 1e9。设备级采样，无内核重放。
用法: python3 analyze_decode.py <sqlite> <t0_s> <t1_s>
"""
import sqlite3, sys, json, statistics

DB = sys.argv[1]
T0 = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
T1 = float(sys.argv[3]) if len(sys.argv) > 3 else 28.5
SCALE = 32.0 / 1e9
ROOF = 246.2  # GB/s 实测

c = sqlite3.connect(DB)
cols = [r[1] for r in c.execute("PRAGMA table_info(TARGET_INFO_GPU_METRICS)")]
idi, nmi = cols.index("metricId"), cols.index("metricName")
names = {r[idi]: r[nmi] for r in c.execute("select * from TARGET_INFO_GPU_METRICS")}

want = {
    "miss": "VidL2 Misses from L1TEX [GB/s]",
    "total": "VidL2 Total from L1TEX [GB/s]",
    "hit": "VidL2 Sector Hit-Rate from L1TEX Reads [Ratio %]",
    "tensor": "SM Tensor Pipe Active [Throughput %]",
    "issue": "SM Issue Stage Throughput [Throughput %]",
    "occ": "Unallocated Warps in Active SMs [Occupancy %]",
    "gr": "GR Engine Active [Throughput %]",
}
ids = {v: k for k, v in names.items()}
tabs = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
gt = next(t for t in tabs if "GPU_METRICS" in t and "TARGET_INFO" not in t)
gcols = [r[1] for r in c.execute(f"PRAGMA table_info({gt})")]
ts = next(x for x in gcols if "imestamp" in x or x == "timestamp")
mid = next(x for x in gcols if "metricId" in x)
val = next(x for x in gcols if x in ("value", "metricValue"))

series = {}
for key, nm in want.items():
    if nm not in ids:
        continue
    rows = list(c.execute(f"select {ts},{val} from {gt} where {mid}=? order by {ts}", (ids[nm],)))
    if not rows:
        continue
    base = rows[0][0]
    series[key] = [((t - base) / 1e9, v) for t, v in rows]

out = {"window_s": [T0, T1], "roof_GB_s": ROOF, "scale_note": "GB/s = raw*32/1e9 (E-DRAM-01 校准)"}
for key, pts in series.items():
    w = [v for t, v in pts if T0 <= t <= T1]
    if not w:
        continue
    if key in ("miss", "total"):
        w = [v * SCALE for v in w]
    s = sorted(w)
    d = {
        "n": len(w), "mean": statistics.fmean(w), "median": statistics.median(w),
        "p90": s[int(len(s) * 0.9)], "max": max(w),
    }
    if key == "miss":
        d["roof_util_mean"] = d["mean"] / ROOF
        d["roof_util_median"] = d["median"] / ROOF
        d["frac_time_above_200GBs"] = sum(1 for v in w if v > 200) / len(w)
        d["frac_time_above_230GBs"] = sum(1 for v in w if v > 230) / len(w)
    out[key] = d

if "miss" in out and "total" in out:
    out["total_over_miss"] = out["total"]["mean"] / out["miss"]["mean"]
print(json.dumps(out, ensure_ascii=False, indent=2))