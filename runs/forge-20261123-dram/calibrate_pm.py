#!/usr/bin/env python3
"""E-DRAM-01 r2 步骤1：校准 nsys PM 的 VidL2 GB/s 口径。

为什么必须校准：本项目已两次被计数器口径坑过——
GB10 无 dram__* 计数器；aperture_device 在统一内存机上恒为 0。
所以在把 "VidL2 Misses from L1TEX [GB/s]" 当 DRAM 流量之前，
必须用**已知字节数**的 streaming 内核核对它的标度。

内存安全（硬约束 §0）：
- 本容器必须带 --memory=3g --memory-swap=3g（外部 docker run 施加）。
- 工作集只用 512 MB（远大于 24 MB L2，足以纯流式；又远小于富余）。
- 不触碰引擎进程，不做任何内核重放。

方法：nsys 后台采样 D 秒；前台跑 reduce 循环 D-3 秒，记 iters 与 elapsed。
已知字节 = iters × buffer_bytes ⇒ 期望 GB/s。
用 "GR Engine Active > 50%" 门控采样点，自动对齐繁忙窗口，避免时间戳基准不一致。
"""
import subprocess, time, json, sys, statistics, sqlite3, os

D = int(sys.argv[1]) if len(sys.argv) > 1 else 12
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/calib"
NBUF = 268435456                      # fp16 元素数 → 512 MiB
BUF_BYTES = NBUF * 2

import torch
x = torch.randn(NBUF, device="cuda", dtype=torch.float16)
torch.cuda.synchronize()

p = subprocess.Popen([
    "nsys", "profile", "--gpu-metrics-devices=all", "--gpu-metrics-set=gb20y-top",
    "--gpu-metrics-frequency=10000", f"--duration={D}",
    "--trace=none", "--sample=none", "--cpuctxsw=none",
    "-o", OUT, "--force-overwrite=true",
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

time.sleep(1.5)
t0 = time.perf_counter()
iters = 0
while time.perf_counter() - t0 < D - 3.5:
    s = x.sum()
    torch.cuda.synchronize()
    iters += 1
elapsed = time.perf_counter() - t0
expected_GBs = iters * BUF_BYTES / elapsed / 1e9
p.wait()

subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite=true",
                "-o", OUT + ".sqlite", OUT + ".nsys-rep"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

c = sqlite3.connect(OUT + ".sqlite")
cols = [r[1] for r in c.execute("PRAGMA table_info(TARGET_INFO_GPU_METRICS)")]
mm = {r[cols.index("metricId")]: r[cols.index("metricName")] for r in
      c.execute("select * from TARGET_INFO_GPU_METRICS")}
gc = [r[1] for r in c.execute("PRAGMA table_info(GPU_METRICS)")]
ti, mi, vi = gc.index("timestamp"), gc.index("metricId"), gc.index("value")
series = {}
for r in c.execute("select * from GPU_METRICS"):
    series.setdefault(mm[r[mi]], {})[r[ti]] = r[vi]

gate_name = "GR Engine Active [Throughput %]"
gate = series.get(gate_name, {})
busy = {t for t, v in gate.items() if v > 50}

res = {"iters": iters, "elapsed_s": elapsed, "buffer_bytes": BUF_BYTES,
       "expected_GBs": expected_GBs, "busy_samples": len(busy),
       "total_samples": len(gate), "metrics": {}}
for name in ["VidL2 Misses from L1TEX [GB/s]", "VidL2 Total from L1TEX [GB/s]",
             "VidL2 Sector Hit-Rate from L1TEX Reads [Ratio %]",
             "SysL2 Throughput [Throughput %]", gate_name]:
    s = series.get(name, {})
    vals = [v for t, v in s.items() if t in busy]
    if not vals:
        continue
    res["metrics"][name] = {"n": len(vals), "mean": statistics.fmean(vals),
                            "median": statistics.median(vals), "max": max(vals)}

m = res["metrics"].get("VidL2 Misses from L1TEX [GB/s]")
if m:
    res["miss_mean_over_expected"] = m["mean"] / expected_GBs
    res["miss_median_over_expected"] = m["median"] / expected_GBs
print(json.dumps(res, indent=2, ensure_ascii=False))
open(OUT + ".json", "w").write(json.dumps(res, indent=2, ensure_ascii=False))