#!/usr/bin/env python3
"""解析 bank 冲突机制闸：确认 padding 真消除了冲突（§7.18 干预有效性单独验证）。"""
import csv, sys

def load(f):
    d = {}
    for r in csv.reader(open(f)):
        if len(r) > 3:
            try: d[r[-3]] = float(r[-1])
            except ValueError: pass
    return d

a, b = load(sys.argv[1]), load(sys.argv[2])
K_ST = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
K_LD = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
K_INS = "smsp__inst_executed_op_shared_st.sum"
K_WF = "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"
K_T = "gpu__time_duration.sum"
print("%-58s %16s %16s %10s" % ("metric", "p0(orig)", "p1(pad)", "delta"))
for k in [K_ST, K_LD, K_INS, K_WF, K_T]:
    x, y = a.get(k, 0), b.get(k, 0)
    r = ("%+.2f%%" % (100 * (y - x) / x)) if x else "n/a"
    print("%-58s %16s %16s %10s" % (k, format(x, ",.0f"), format(y, ",.0f"), r))
sa, sb = a.get(K_INS, 1), b.get(K_INS, 1)
print()
print("store conflict multiple: p0 %.3fx -> p1 %.3fx" % (a.get(K_ST, 0) / sa, b.get(K_ST, 0) / sb))
print("shared wavefront delta: %+.2f%%" % (100 * (b.get(K_WF, 0) - a.get(K_WF, 0)) / a.get(K_WF, 1)))
print("kernel time delta:      %+.2f%%" % (100 * (b.get(K_T, 0) - a.get(K_T, 0)) / a.get(K_T, 1)))