#!/usr/bin/env python3
"""解析 E-SELF-08b：v1 vs mma2 的机制对比，判决木桶分析是否成立。"""
import csv, os

os.chdir(os.path.expanduser("~/ptxwork"))
ROOF = 246.2  # GB/s，E-DRAM-01 两次校准


def load(f):
    lines = open(f).read().splitlines()
    s = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    d, kn = {}, ""
    for r in csv.DictReader(lines[s:]):
        kn = r.get("Kernel Name", kn)
        try:
            d[r["Metric Name"]] = float(r["Metric Value"].replace(",", ""))
        except (ValueError, KeyError):
            pass
    return d, kn


a, ka = load("m2-v1.csv")
b, kb = load("m2-mma2.csv")
assert a and b, "CSV 解析为空"
print("v1   内核:", ka[:70])
print("mma2 内核:", kb[:70])
print()

print("=== 占用率与其限制资源 ===")
print(f"{'量':<50}{'v1':>14}{'mma2':>14}")
for k in ("launch__registers_per_thread", "launch__shared_mem_per_block_static",
          "launch__shared_mem_per_block_dynamic", "launch__occupancy_limit_registers",
          "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_warps",
          "sm__warps_active.avg.pct_of_peak_sustained_active"):
    print(f"{k:<50}{a.get(k,0):>14.2f}{b.get(k,0):>14.2f}")

print()
print("=== 供给吞吐 ===")
for k in ("sm__throughput.avg.pct_of_peak_sustained_elapsed",
          "lts__throughput.avg.pct_of_peak_sustained_elapsed",
          "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
          "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"):
    print(f"{k:<50}{a.get(k,0):>14.2f}{b.get(k,0):>14.2f}")

print()
print("=== 判决读数：达成 DRAM 带宽 ===")
rows = []
for tag, d in (("v1", a), ("mma2", b)):
    sec = d["gpu__time_duration.sum"] / 1e9
    rd = d["lts__t_sectors_op_read.sum"] * 32
    miss = d["lts__t_sectors_op_read_lookup_miss.sum"] * 32
    bw = miss / sec / 1e9
    rows.append((tag, sec, rd, miss, bw))
    print(f"{tag:5s} 时长 {sec*1e3:8.4f} ms  L2读请求 {rd/2**30:6.3f} GiB ({rd/sec/1e9:6.1f} GB/s)"
          f"  miss {miss/2**30:6.3f} GiB  达成 {bw:6.1f} GB/s = 屋顶 {100*bw/ROOF:5.1f}%"
          f"  L2命中 {100*(1-miss/rd):5.1f}%")
if len(rows) == 2:
    (_, s1, _, m1, w1), (_, s2, _, m2, w2) = rows
    print(f"\n  带宽变化 {w1:.1f} → {w2:.1f} GB/s ({100*(w2-w1)/w1:+.1f}%)")
    print(f"  时长变化 {s1*1e3:.4f} → {s2*1e3:.4f} ms ({100*(s2-s1)/s1:+.1f}%)")
    print(f"  miss 字节变化 {m1/2**30:.3f} → {m2/2**30:.3f} GiB ({100*(m2-m1)/m1:+.1f}%)")
    verdict = "成立：更低占用率下带宽反而更高 ⇒ 卡点是在飞字节数，不是 warp 数" if w2 > w1 * 1.02 else \
              ("不成立：带宽未上升 ⇒ 需另找卡点" if w2 < w1 * 0.98 else "带宽基本持平，机制未分离")
    print(f"  判决：{verdict}")

print()
print("=== warp 延迟分解（只用 .pct 归一）===")
S = [k for k in a if "issue_stalled" in k and k.endswith(".pct")]
ta = sum(a[k] for k in S)
tb = sum(b.get(k, 0) for k in S)
print(f"{'原因':<24}{'v1':>10}{'占比':>9}{'mma2':>10}{'占比':>9}")
for k in sorted(S, key=lambda k: -a[k]):
    r = k.replace("smsp__average_warp_latency_issue_stalled_", "").replace(".pct", "")
    va, vb = a[k], b.get(k, 0)
    print(f"{r:<24}{va:>10.0f}{100*va/ta:>8.1f}%{vb:>10.0f}{100*vb/tb:>8.1f}%")

print()
print("=== 指令成分（cp.async 应减少 shared_st / 改变 global_ld）===")
for k in ("smsp__inst_executed.sum", "smsp__inst_executed_op_global_ld.sum",
          "smsp__inst_executed_op_shared_st.sum", "smsp__inst_executed_op_ldsm.sum",
          "l1tex__t_sectors_pipe_lsu_mem_global_op_ldgsts_cache_access.sum"):
    va, vb = a.get(k, 0), b.get(k, 0)
    print(f"{k:<62}{va:>16,.0f}{vb:>16,.0f}")