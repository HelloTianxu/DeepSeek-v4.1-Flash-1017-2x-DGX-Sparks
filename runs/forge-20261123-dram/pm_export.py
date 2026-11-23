#!/usr/bin/env python3
"""E-DRAM-01 r2：解析 nsys GPU PM 采样的 sqlite。

用法：
  python3 pm_export.py list <sqlite>                 列出所有计数器名
  python3 pm_export.py series <sqlite> <out.json>    导出全部计数器的时间序列摘要
  python3 pm_export.py window <sqlite> <t0> <t1>     指定时间窗内各计数器统计

口径说明：PM 采样是**设备级**、无内核重放、不注入引擎进程。
它给的是整个 GPU 在采样窗口内的吞吐，**不是单个内核的字节数**，
因此窗口必须与引擎自身相位时间对齐后才能归因到专家核心。
"""
import sqlite3, sys, json, statistics


def tables(c):
    return [r[0] for r in c.execute("select name from sqlite_master where type='table'")]


def metric_map(c):
    cols = [r[1] for r in c.execute("PRAGMA table_info(TARGET_INFO_GPU_METRICS)")]
    rows = list(c.execute("select * from TARGET_INFO_GPU_METRICS"))
    # 找 id 列与 name 列
    idi = cols.index("metricId")
    nmi = cols.index("metricName")   # 精确匹配：typeName 也含 "name"，会误命中
    assert nmi != idi
    return cols, {r[idi]: r[nmi] for r in rows}


def gpu_cols(c):
    return [r[1] for r in c.execute("PRAGMA table_info(GPU_METRICS)")]


def main():
    mode, path = sys.argv[1], sys.argv[2]
    c = sqlite3.connect(path)
    cols, mm = metric_map(c)
    gc = gpu_cols(c)

    if mode == "list":
        print("TARGET_INFO_GPU_METRICS cols:", cols)
        print("GPU_METRICS cols:", gc)
        print("metric count:", len(mm))
        for k in sorted(mm):
            print(f"  {k}\t{mm[k]}")
        return

    idi = next(i for i, n in enumerate(gc) if "metricId" in n)
    tsi = next(i for i, n in enumerate(gc) if "timestamp" in n.lower())
    vai = next(i for i, n in enumerate(gc) if n.lower() == "value")

    rows = list(c.execute("select * from GPU_METRICS"))
    series = {}
    for r in rows:
        series.setdefault(r[idi], []).append((r[tsi], r[vai]))

    if mode == "series":
        out = {}
        for mid, pts in series.items():
            pts.sort()
            vals = [v for _, v in pts]
            out[mm.get(mid, str(mid))] = {
                "metric_id": mid,
                "n": len(pts),
                "t_first_ns": pts[0][0],
                "t_last_ns": pts[-1][0],
                "min": min(vals), "max": max(vals),
                "mean": statistics.fmean(vals),
                "median": statistics.median(vals),
            }
        open(sys.argv[3], "w").write(json.dumps(out, indent=2, ensure_ascii=False))
        print(json.dumps(out, indent=2, ensure_ascii=False)[:3000])
        return

    if mode == "window":
        t0, t1 = int(sys.argv[3]), int(sys.argv[4])
        res = {}
        for mid, pts in series.items():
            vals = [v for t, v in pts if t0 <= t <= t1]
            if not vals:
                continue
            res[mm.get(mid, str(mid))] = {
                "n": len(vals), "min": min(vals), "max": max(vals),
                "mean": statistics.fmean(vals), "median": statistics.median(vals),
            }
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    raise SystemExit("mode must be list|series|window")


if __name__ == "__main__":
    main()