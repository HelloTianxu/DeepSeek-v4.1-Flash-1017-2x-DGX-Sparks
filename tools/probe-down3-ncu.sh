#!/usr/bin/env bash
# E-SELF-08d：down 行 v1 vs down3 的机制读数（生产形状）。
set -euo pipefail
cd "$(dirname "$0")"

M="gpu__time_duration.sum"
M="${M},lts__t_sectors_op_read.sum,lts__t_sectors_op_read_lookup_miss.sum"
M="${M},lts__t_sectors_op_write.sum"
M="${M},lts__throughput.avg.pct_of_peak_sustained_elapsed"
M="${M},l1tex__throughput.avg.pct_of_peak_sustained_elapsed"
M="${M},sm__throughput.avg.pct_of_peak_sustained_elapsed"
M="${M},sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
M="${M},sm__warps_active.avg.pct_of_peak_sustained_active"
M="${M},launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem"
M="${M},launch__registers_per_thread"
M="${M},launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic"
M="${M},smsp__inst_executed.sum,smsp__inst_executed_op_global_ld.sum"
M="${M},smsp__inst_executed_op_shared_st.sum,smsp__inst_executed_op_ldsm.sum"
for r in barrier wait long_scoreboard short_scoreboard mio_throttle lg_throttle math_pipe_throttle not_selected selected; do
  M="${M},smsp__average_warp_latency_issue_stalled_${r}"
done

for A in v1 down3; do
  case "$A" in
    v1)    KN="regex:grouped_rows_kernel" ;;
    down3) KN="regex:grouped_down3_kernel" ;;
  esac
  echo "=== $A ==="
  ncu --clock-control none --kernel-name "$KN" --launch-skip 3 --launch-count 1 \
      --metrics "$M" --csv --log-file "d3-${A}.csv" \
      python3 probe-down3.py --only "$A" --ncu-warm > "d3-${A}.log" 2>&1 || {
        echo "NCU_FAILED $A"; tail -20 "d3-${A}.log"; exit 1; }
  test -s "d3-${A}.csv" || { echo "EMPTY_CSV $A"; exit 1; }
  echo "lines: $(wc -l < "d3-${A}.csv")"
done
echo D3NCU_OK