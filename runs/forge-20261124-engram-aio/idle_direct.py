#!/usr/bin/env python3
"""不依赖分段的 GPU 空闲口径：直接数 prefill 窗口内 GR Engine Active <= 20% 的采样。

分段法（analyze_prod.py）会把 <=2 个采样间隔（400us）的空隙并入区段，
故 "prefill - 各区段时长之和" 不是干净的空闲量。本脚本按采样点直接统计，
并对最长连续空闲段排序，回答「Engram 卡点是否真的疏解」。

用法: python3 idle_direct.py <sqlite> <label>
"""
import sqlite3, sys, statistics

PATH, LABEL = sys.argv[1], sys.argv[2]
c = sqlite3.connect(PATH)

# 取 GR Engine Active 的时间序列（metricName 精确匹配，避免误命中 typeName）
q = """
SELECT s.timestamp, s.value
FROM GPU_METRICS s JOIN TARGET_INFO_GPU_METRICS m ON s.metricId = m.metricId
WHERE m.metricName = 'GR Engine Active [Throughput %]'
ORDER BY s.timestamp
"""
rows = c.execute(q).fetchall()
if not rows:
    raise SystemExit("no GR Engine Active series")

dt = statistics.median([b[0] - a[0] for a, b in zip(rows, rows[1:])])
busy = [(t, v) for t, v in rows if v > 20]
t0, t1 = busy[0][0], busy[-1][0]
win = [(t, v) for t, v in rows if t0 <= t <= t1]

idle = [(t, v) for t, v in win if v <= 20]
span_s = (t1 - t0) / 1e9
idle_s = len(idle) * dt / 1e9

# 最长连续空闲段（允许 1 个采样间隔抖动）
runs, cur = [], []
prev = None
for t, v in win:
    if v <= 20:
        if prev is not None and t - prev > 2 * dt:
            if cur: runs.append(cur); cur = []
        cur.append(t)
    else:
        if cur: runs.append(cur); cur = []
    prev = t
if cur: runs.append(cur)
runs = sorted(((len(r) * dt / 1e6, r[0]) for r in runs), reverse=True)

print(f"[{LABEL}] 活跃窗口 {span_s:.4f}s  采样 {len(win)}  间隔 {dt/1000:.1f}us")
print(f"  GR<=20% 采样 {len(idle)} 个 => 空闲 {idle_s:.4f}s ({100*idle_s/span_s:.2f}%)")
print(f"  连续空闲段数 {len(runs)}；最长 5 段(ms): {[round(x[0],2) for x in runs[:5]]}")
print(f"  >10ms 的空闲段: {sum(1 for x in runs if x[0]>10)} 个，"
      f"合计 {sum(x[0] for x in runs if x[0]>10)/1000:.4f}s")