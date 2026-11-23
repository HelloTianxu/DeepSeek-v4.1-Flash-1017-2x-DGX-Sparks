#!/usr/bin/env python3
"""E-DRAM-01 r2 步骤4：用 Tensor Pipe 活跃度定位 MMA 相位，再读该相位的 DRAM 带宽。

为什么换门控：步骤3 用 "DRAM>150GB/s" 找段，发现这些段的 Tensor Active 中位数为 0
⇒ 高带宽时刻不是专家 MMA（更像搬运/attention/Engram）。
专家核心的判据应该是 **Tensor 管线在跑**，而不是"带宽高"。
本脚本按 Tensor Active 门控，避免用结论去选样本（循环论证）。

单位：GB/s = raw × 32 / 1e9（已校准）。
边界：设备级采样，Tensor 相位里除专家还含 attention/HC 的 MMA，
故同时报告段长分布——专家 gateup≈14ms、down≈8ms 是可识别的特征长度。
"""
import sqlite3, sys, json, statistics

PATH = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/tensor-phase.json"
S = 32.0 / 1e9

c = sqlite3.connect(PATH)
cols = [r[1] for r in c.execute("PRAGMA table_info(TARGET_INFO_GPU_METRICS)")]
mm = {r[cols.index("metricId")]: r[cols.index("metricName")]
      for r in c.execute("select * from TARGET_INFO_GPU_METRICS")}
gc = [r[1] for r in c.execute("PRAGMA table_info(GPU_METRICS)")]
ti, mi, vi = gc.index("timestamp"), gc.index("metricId"), gc.index("value")
want = ["GR Engine Active [Throughput %]", "SM Tensor Pipe Active [Throughput %]",
        "VidL2 Misses from L1TEX [GB/s]", "VidL2 Total from L1TEX [GB/s]",
        "VidL2 Sector Hit-Rate from L1TEX Reads [Ratio %]",
        "SM Issue Stage Throughput [Throughput %]",
        "Sync CS SM Warps [Avg Warps]", "Sync CS SM Warps [Occupancy %]",
        "SM ALU Pipe Throughput [Throughput %]", "SM FMA Pipe Throughput [Throughput %]",
        "SM XU Pipe Throughput [Throughput %]", "L1TEX Throughput [Throughput %]"]
series = {}
for r in c.execute("select * from GPU_METRICS"):
    n = mm.get(r[mi])
    if n in want:
        series.setdefault(n, {})[r[ti]] = r[vi]

ten = series["SM Tensor Pipe Active [Throughput %]"]
miss = series["VidL2 Misses from L1TEX [GB/s]"]
gate = series["GR Engine Active [Throughput %]"]
ts_all = sorted(ten)
dt = statistics.median([b - a for a, b in zip(ts_all, ts_all[1:])])

# 只取最长的那个 chunk（与步骤3 一致：约 940ms）
busy = sorted(t for t, v in gate.items() if v > 20)
regions, cur = [], [busy[0]]
for a, b in zip(busy, busy[1:]):
    if b - a <= 2 * dt:
        cur.append(b)
    else:
        regions.append(cur); cur = [b]
regions.append(cur)
reg = max(regions, key=lambda r: r[-1] - r[0])
tset = set(reg)


def runs(pred, min_ms=1.0):
    out, cur = [], []
    for t in sorted(tset):
        if pred(t):
            cur.append(t)
        else:
            if cur and (cur[-1] - cur[0]) >= min_ms * 1e6:
                out.append(cur)
            cur = []
    if cur and (cur[-1] - cur[0]) >= min_ms * 1e6:
        out.append(cur)
    return out


def summarize(ts):
    st = set(ts)
    o = {"n": len(ts)}
    for n in want:
        vals = [v for t, v in series.get(n, {}).items() if t in st]
        if not vals:
            continue
        if n.endswith("[GB/s]"):
            vals = [v * S for v in vals]
        o[n] = {"mean": statistics.fmean(vals), "median": statistics.median(vals),
                "max": max(vals)}
    return o


res = {"chunk_ms": (reg[-1] - reg[0]) / 1e6, "sample_interval_ns": dt, "gates": {}}
for th in (10, 20, 40, 60):
    rs = runs(lambda t: ten.get(t, 0) > th)
    if not rs:
        continue
    durs = [(r[-1] - r[0]) / 1e6 for r in rs]
    allts = [t for r in rs for t in r]
    res["gates"][f"tensor_gt_{th}"] = {
        "n_segments": len(rs),
        "total_ms": sum(durs),
        "frac_of_chunk": sum(durs) / ((reg[-1] - reg[0]) / 1e6),
        "seg_ms": {"mean": statistics.fmean(durs), "median": statistics.median(durs),
                   "max": max(durs)},
        "longest_12": [{"ms": (r[-1] - r[0]) / 1e6,
                        "DRAM_GBs_mean": statistics.fmean(
                            [miss[t] * S for t in r if t in miss])}
                       for r in sorted(rs, key=lambda r: r[0] - r[-1])[:12]],
        "pooled": summarize(allts),
    }
open(OUT, "w").write(json.dumps(res, indent=2, ensure_ascii=False))
print(json.dumps(res, indent=2, ensure_ascii=False))