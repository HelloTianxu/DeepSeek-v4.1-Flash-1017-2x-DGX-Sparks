#!/usr/bin/env python3
"""解析 E-SELF-07 的 ncu CSV：按因果链顺序打印 warp 延迟分解与各级供给饱和度。"""
import csv, sys, os

os.chdir(os.path.expanduser("~/ptxwork"))


def load(f):
    # ncu 的 --log-file 会在 CSV 前写 ==WARNING== / ==PROF== 前言，必须跳到真正的表头
    lines = open(f).read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    d = {}
    for r in csv.DictReader(lines[start:]):
        n = r.get("Metric Name", "")
        v = r.get("Metric Value", "")
        try:
            d[n] = float(v.replace(",", ""))
        except ValueError:
            pass
    return d


a, b = load("ncu-mma.csv"), load("ncu-rows.csv")
assert a and b, "CSV 解析为空"
S = [k for k in a if "issue_stalled" in k and k.endswith(".pct")]
assert len(S) >= 15, f"stall 指标不足: {len(S)}"
ta = sum(a[k] for k in S)
tb = sum(b.get(k, 0) for k in S)

print("指标数 mma", len(a), " rows", len(b))
print()
hdr = "warp 延迟原因"
print(f"{hdr:<24}{'gate/up':>11}{'占比':>9}   {'down':>11}{'占比':>9}")
print("-" * 68)
for k in sorted(S, key=lambda k: -a[k]):
    r = k.replace("smsp__average_warp_latency_issue_stalled_", "")
    va, vb = a[k], b.get(k, 0)
    print(f"{r:<24}{va:>11.3f}{100*va/ta:>8.1f}%   {vb:>11.3f}{100*vb/tb:>8.1f}%")
print("-" * 68)
print(f"{'合计 warp cycles/inst':<24}{ta:>11.3f}{100.0:>8.1f}%   {tb:>11.3f}{100.0:>8.1f}%")
print()
print("=== 各级供给吞吐 / 饱和度（左 gate/up，右 down）===")
for k in sorted(a):
    if "issue_stalled" in k:
        continue
    print(f"{k:<62}{a[k]:>16.2f}{b.get(k,0):>16.2f}")
print()
print("=== 派生量（sector=32B，口径经 E-DRAM-01 两次校准）===")
for tag, d in (("gate/up", a), ("down", b)):
    dur_ns = d.get("gpu__time_duration.sum", 0)
    sec = dur_ns / 1e9
    rd = d.get("lts__t_sectors_op_read.sum", 0) * 32
    miss = d.get("lts__t_sectors_op_read_lookup_miss.sum", 0) * 32
    wr = d.get("lts__t_sectors_op_write.sum", 0) * 32
    sh = d.get("l1tex__data_pipe_lsu_wavefronts_mem_shared.sum", 0)
    print(f"{tag}: 时长 {sec*1e3:.4f} ms")
    if sec > 0:
        print(f"   L2 读请求 {rd/2**30:.3f} GiB → {rd/sec/1e9:.1f} GB/s")
        print(f"   L2 读 miss {miss/2**30:.3f} GiB → {miss/sec/1e9:.1f} GB/s (DRAM 上界)")
        print(f"   L2 读命中 {100*(1-miss/rd) if rd else 0:.1f}%")
        print(f"   L2 写 {wr/2**30:.3f} GiB → {wr/sec/1e9:.1f} GB/s")
        print(f"   shared wavefronts {sh/1e6:.2f} M")