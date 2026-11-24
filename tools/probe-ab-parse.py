#!/usr/bin/env python3
"""E-SELF-08：两臂机制对比的统一解析器（带正确标签，避免存档列名错标）。

用法：probe-ab-parse.py --a-file X.csv --b-file Y.csv --a-label v1 --b-label mma2
判决口径：DRAM miss 字节几乎不变而达成带宽上升 ⇒ 卡口在片内递货，不在 DRAM 容量。
"""
import csv, os, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--a-file", required=True)
ap.add_argument("--b-file", required=True)
ap.add_argument("--a-label", required=True)
ap.add_argument("--b-label", required=True)
ap.add_argument("--roof", type=float, default=246.2, help="DRAM 可持续读屋顶 GB/s（E-DRAM-01 两次校准）")
args = ap.parse_args()
A, B = args.a_label, args.b_label
os.chdir(os.path.expanduser("~/ptxwork"))


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


a, ka = load(args.a_file)
b, kb = load(args.b_file)
assert a and b, "CSV 解析为空，禁止在此情况下做任何判定"
print(f"{A:6s} 内核: {ka[:76]}")
print(f"{B:6s} 内核: {kb[:76]}\n")

W = max(len(A), len(B)) + 2
print("=== 占用率与其限制资源 ===")
print(f"{'量':<52}{A:>{W+10}}{B:>{W+10}}")
for k in ("launch__registers_per_thread", "launch__shared_mem_per_block_static",
          "launch__shared_mem_per_block_dynamic", "launch__occupancy_limit_registers",
          "launch__occupancy_limit_shared_mem",
          "sm__warps_active.avg.pct_of_peak_sustained_active"):
    print(f"{k:<52}{a.get(k,0):>{W+10}.2f}{b.get(k,0):>{W+10}.2f}")

print("\n=== 供给吞吐 ===")
for k in ("sm__throughput.avg.pct_of_peak_sustained_elapsed",
          "lts__throughput.avg.pct_of_peak_sustained_elapsed",
          "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
          "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"):
    print(f"{k:<52}{a.get(k,0):>{W+10}.2f}{b.get(k,0):>{W+10}.2f}")

print("\n=== 判决读数：达成 DRAM 带宽 ===")
rows = []
for tag, d in ((A, a), (B, b)):
    sec = d["gpu__time_duration.sum"] / 1e9
    rd = d["lts__t_sectors_op_read.sum"] * 32
    miss = d["lts__t_sectors_op_read_lookup_miss.sum"] * 32
    bw = miss / sec / 1e9
    rows.append((sec, miss, bw))
    print(f"{tag:6s} 时长 {sec*1e3:8.4f} ms  L2读请求 {rd/2**30:6.3f} GiB ({rd/sec/1e9:6.1f} GB/s)"
          f"  miss {miss/2**30:6.3f} GiB  达成 {bw:6.1f} GB/s = 屋顶 {100*bw/args.roof:5.1f}%"
          f"  L2命中 {100*(1-miss/rd):5.1f}%")
(s1, m1, w1), (s2, m2, w2) = rows
print(f"\n  带宽 {w1:.1f} → {w2:.1f} GB/s ({100*(w2-w1)/w1:+.1f}%)")
print(f"  时长 {s1*1e3:.4f} → {s2*1e3:.4f} ms ({100*(s2-s1)/s1:+.1f}%)")
print(f"  miss 字节 {m1/2**30:.3f} → {m2/2**30:.3f} GiB ({100*(m2-m1)/m1:+.1f}%)")
occ_a = a.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0)
occ_b = b.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0)
print(f"  占用率 {occ_a:.2f}% → {occ_b:.2f}% ({'升' if occ_b>occ_a else '降'})")
if w2 > w1 * 1.02 and abs(m2 - m1) / m1 < 0.05:
    print(f"  判决：成立 —— DRAM 字节几乎不变而带宽上升，且占用率{'下降' if occ_b<occ_a else '上升'}，"
          f"故卡口是片内递货（在飞字节/发射槽），不是 warp 数、不是 DRAM 容量")
else:
    print("  判决：未成立或机制未分离，需另找卡口")

print("\n=== warp 延迟分解（只用 .pct 归一，不混 .ratio）===")
S = [k for k in a if "issue_stalled" in k and k.endswith(".pct")]
ta = sum(a[k] for k in S); tb = sum(b.get(k, 0) for k in S)
assert ta > 0 and tb > 0, "stall 合计为零，读数不可用"
print(f"{'原因':<24}{A:>10}{'占比':>9}{B:>10}{'占比':>9}")
for k in sorted(S, key=lambda k: -a[k]):
    r = k.replace("smsp__average_warp_latency_issue_stalled_", "").replace(".pct", "")
    va, vb = a[k], b.get(k, 0)
    print(f"{r:<24}{va:>10.0f}{100*va/ta:>8.1f}%{vb:>10.0f}{100*vb/tb:>8.1f}%")

print("\n=== 指令成分 ===")
print(f"{'量':<62}{A:>16}{B:>16}")
for k in ("smsp__inst_executed.sum", "smsp__inst_executed_op_global_ld.sum",
          "smsp__inst_executed_op_shared_st.sum", "smsp__inst_executed_op_ldsm.sum"):
    print(f"{k:<62}{a.get(k,0):>16,.0f}{b.get(k,0):>16,.0f}")
print("注：l1tex ldgsts_cache_access 在 GB10 对 cp.async 恒报 0，不可据此断言无 cp.async（§7.10）")