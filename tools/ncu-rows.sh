#!/usr/bin/env bash
# E-ROWS-01 带宽采集：验证 R32 是否按 MLP 模型把有效带宽推到 ~175 GB/s。
# 判决不只测时间，必须测中介变量（手册 §7.32）。
# metric 集与 E-OCC-02/03 完全一致，保证口径单一、可直接对比。
set -uo pipefail
cd /root/ptxwork

METRICS="gpc__cycles_elapsed.avg,\
l1tex__m_l1tex2xbar_throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_sectors_op_read.sum,\
lts__t_sectors_op_read_lookup_hit.sum,\
lts__t_sectors_op_read_lookup_miss.sum,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__inst_executed.sum,\
smsp__issue_active.avg.pct_of_peak_sustained_active"

for arm in "$@"; do
  echo "=== ncu $arm ==="
  # -k regex 必须给：否则会抓到 group_count 等小 kernel（手册 §7.26）
  ncu --csv --log-file "rows-${arm}.csv" \
      --metrics "$METRICS" \
      -k "regex:grouped_mma_kernel" -c 1 -s 3 \
      python3 probe-rows.py --only "$arm" 2>&1 | tail -4
  echo "  -> rows-${arm}.csv"
done