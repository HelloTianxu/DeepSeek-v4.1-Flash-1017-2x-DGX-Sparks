#!/usr/bin/env python3
"""E-DRAM-01 r2 步骤3：分析生产 prefill 期间的设备级 DRAM 吞吐。

单位口径（已两次校准，见 calib-r1.json / calib-5k.json）：
  sqlite 中标着 [GB/s] 的 VidL2 字段，原始值其实是 **sectors/s**；
  GB/s = raw × 32 / 1e9。10 kHz 误差 0.18%，5 kHz 误差 0.24%，与频率无关。

判决逻辑：
  专家核心占 kernel 窗口 38.38%（既有生产 trace）。
  若 prefill 繁忙期内存在占时约该量级、 level 贴近实测 DRAM 屋顶 246 GB/s 的平台
    ⇒ 专家核心实际流量 ≈ 2× 唯一字节且已在屋顶（H1 tile 重读）。
  若主导平台只有约 123 GB/s
    ⇒ 123 GB/s 就是真实流量，距屋顶一倍（H2 内核效率）。
  VidL2 Total（L2 请求）与 Misses（DRAM）之比进一步区分"重读被 L2 吸收"。

边界：PM 采样是**整个 GPU**的吞吐，不是单内核字节数；
attention/HC/Engram/NCCL 与专家核心混在同一窗口，故必须看分段结构而非全窗均值。
"""
import sqlite3, sys, json, statistics

PATH = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/pm-prod-analysis.json"
SCALE = 32.0 / 1e9          # sectors/s -> GB/s

c = sqlite3.connect(PATH)
cols = [r[1] for r in c.execute("PRAGMA table_info(TARGET_INFO_GPU_METRICS)")]
mm = {r[cols.index("metricId")]: r[cols.index("metricName")]
      for r in c.execute("select * from TARGET_INFO_GPU_METRICS")}
gc = [r[1] for r in c.execute("PRAGMA table_info(GPU_METRICS)")]
ti, mi, vi = gc.index("timestamp"), gc.index("metricId"), gc.index("value")

want = {
    "GR Engine Active [Throughput %]",
    "VidL2 Misses from L1TEX [GB/s]",
    "VidL2 Total from L1TEX [GB/s]",
    "VidL2 Sector Hit-Rate from L1TEX Reads [Ratio %]",
    "SM Tensor Pipe Active [Throughput %]",
    "SM Issue Stage Throughput [Throughput %]",
    "Sync CS SM Warps [Avg Warps]",
    "Sync CS SM Warps [Occupancy %]",
    "Async CS SM Warps [Avg Warps]",
    "SysL2 Throughput [Throughput %]",
    "GPC Frequency [MHz]",
}
series = {}
for r in c.execute("select * from GPU_METRICS"):
    n = mm.get(r[mi])
    if n in want:
        series.setdefault(n, {})[r[ti]] = r[vi]

gate = series["GR Engine Active [Throughput %]"]
miss = series["VidL2 Misses from L1TEX [GB/s]"]
tot = series["VidL2 Total from L1TEX [GB/s]"]

ts = sorted(t for t, v in gate.items() if v > 20)
if not ts:
    raise SystemExit("no busy samples")
dt = statistics.median([b - a for a, b in zip(ts, ts[1:])])

# 连续繁忙区段（允许 2 个采样间隔的间隙）
regions, cur = [], [ts[0]]
for a, b in zip(ts, ts[1:]):
    if b - a <= 2 * dt:
        cur.append(b)
    else:
        regions.append(cur); cur = [b]
regions.append(cur)
regions = [r for r in regions if (r[-1] - r[0]) > 50e6]   # >50ms 才算一段


def stats(names, tset):
    out = {}
    for n in names:
        s = series.get(n, {})
        vals = [v for t, v in s.items() if t in tset]
        if vals:
            out[n] = {"n": len(vals), "mean": statistics.fmean(vals),
                      "median": statistics.median(vals), "max": max(vals)}
    return out


res = {"scale_note": "GB/s = raw * 32 / 1e9 (calibrated)",
       "sample_interval_ns": dt, "busy_regions": []}

for reg in regions:
    tset = set(reg)
    dur_s = (reg[-1] - reg[0]) / 1e9
    pts = sorted((t, miss[t] * SCALE) for t in reg if t in miss)
    if not pts:
        continue
    vals = [v for _, v in pts]
    hist = {}
    for v in vals:
        b = int(v // 20) * 20
        hist[f"{b}-{b+20}"] = hist.get(f"{b}-{b+20}", 0) + 1
    # 高带宽连续段
    segs, cur2 = [], []
    for t, v in pts:
        if v >= 150:
            cur2.append((t, v))
        else:
            if cur2:
                segs.append(cur2); cur2 = []
    if cur2:
        segs.append(cur2)
    segs = [s for s in segs if (s[-1][0] - s[0][0]) > 1e6]   # >1ms
    segs.sort(key=lambda s: s[-1][0] - s[0][0], reverse=True)
    hi_t = {t for s in segs for t, _ in s}
    res["busy_regions"].append({
        "t0_ns": reg[0], "t1_ns": reg[-1], "duration_s": dur_s,
        "n_samples": len(pts),
        "DRAM_GBs": {"mean": statistics.fmean(vals), "median": statistics.median(vals),
                     "max": max(vals), "p90": sorted(vals)[int(len(vals) * 0.9)]},
        "histogram_GBs": dict(sorted(hist.items(), key=lambda kv: -kv[1])[:12]),
        "frac_time_above_150GBs": len(hi_t) / len(pts),
        "n_segments_above_150": len(segs),
        "top_segments": [{"ms": (s[-1][0] - s[0][0]) / 1e6,
                          "mean_GBs": statistics.fmean([v for _, v in s])} for s in segs[:12]],
        "during_high_segments": stats(
            ["VidL2 Total from L1TEX [GB/s]", "VidL2 Sector Hit-Rate from L1TEX Reads [Ratio %]",
             "SM Tensor Pipe Active [Throughput %]", "SM Issue Stage Throughput [Throughput %]",
             "Sync CS SM Warps [Avg Warps]", "Sync CS SM Warps [Occupancy %]"], hi_t),
        "whole_region": stats(
            ["VidL2 Total from L1TEX [GB/s]", "SM Tensor Pipe Active [Throughput %]",
             "Sync CS SM Warps [Avg Warps]", "Sync CS SM Warps [Occupancy %]",
             "SysL2 Throughput [Throughput %]", "GPC Frequency [MHz]"], tset),
    })

# Total 与 Misses 的全局比（重读是否被 L2 吸收）
for r in res["busy_regions"]:
    w = r["whole_region"].get("VidL2 Total from L1TEX [GB/s]")
    if w:
        r["total_over_miss_mean"] = (w["mean"] * SCALE) / r["DRAM_GBs"]["mean"]
        r["total_mean_GBs"] = w["mean"] * SCALE

open(OUT, "w").write(json.dumps(res, indent=2, ensure_ascii=False))
print(json.dumps(res, indent=2, ensure_ascii=False))