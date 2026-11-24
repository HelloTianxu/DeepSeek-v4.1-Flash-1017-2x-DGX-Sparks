#!/usr/bin/env python3
"""解析 E-SELF-07b：管线饱和度与指令成分，回答「造币车间是否已满负荷」。"""
import csv, os

os.chdir(os.path.expanduser("~/ptxwork"))


def load(f):
    lines = open(f).read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    d = {}
    for r in csv.DictReader(lines[start:]):
        try:
            d[r["Metric Name"]] = float(r["Metric Value"].replace(",", ""))
        except (ValueError, KeyError):
            pass
    return d


a, b = load("pipe-mma.csv"), load("pipe-rows.csv")
assert a and b, "CSV 解析为空"

print("=== 执行管线饱和度（% of peak sustained active）===")
print(f"{'管线':<52}{'gate/up':>10}{'down':>10}")
for k in sorted(a):
    if "cycles_active" in k or "issue_active" in k or "inst_issued" in k:
        print(f"{k:<52}{a[k]:>10.2f}{b.get(k,0):>10.2f}")

print()
print("=== 指令成分（warp 指令条数）===")
tot_a = a.get("smsp__inst_executed.sum", 0)
tot_b = b.get("smsp__inst_executed.sum", 0)
print(f"{'类别':<52}{'gate/up':>14}{'占比':>8}{'down':>14}{'占比':>8}")
for k in sorted(a):
    if "inst_executed_op" in k:
        va, vb = a[k], b.get(k, 0)
        pa = 100 * va / tot_a if tot_a else 0
        pb = 100 * vb / tot_b if tot_b else 0
        print(f"{k.replace('smsp__inst_executed_op_',''):<52}{va:>14,.0f}{pa:>7.2f}%{vb:>14,.0f}{pb:>7.2f}%")
print(f"{'总 warp 指令':<52}{tot_a:>14,.0f}{100:>7.2f}%{tot_b:>14,.0f}{100:>7.2f}%")

print()
print("=== shared bank 冲突 与 占用率 ===")
for k in sorted(a):
    if "bank_conflicts" in k or "warps_active" in k:
        print(f"{k:<62}{a[k]:>16,.2f}{b.get(k,0):>16,.2f}")

# MMA 指令数推算：执行行数 × N × K × 2 FLOP / 每条 m16n8k16 的 4096 FLOP
print()
print("=== MMA 占指令流的比例（按执行行数 17056 推算）===")
EX_ROWS = 17056
for tag, d, N, K, mats in (("gate/up", a, 1152, 5120, 2), ("down", b, 5120, 1152, 1)):
    flop = 2 * EX_ROWS * N * mats * K
    mma = flop / 4096
    t = d.get("smsp__inst_executed.sum", 0)
    print(f"{tag}: 执行 FLOP {flop/1e9:.1f} G  MMA 指令 ≈{mma/1e6:.1f} M  "
          f"占总指令 {100*mma/t if t else 0:.2f}%  → 非 MMA 占 {100-100*mma/t if t else 0:.2f}%")