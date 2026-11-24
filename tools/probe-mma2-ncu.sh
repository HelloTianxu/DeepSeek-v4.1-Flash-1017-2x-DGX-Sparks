#!/usr/bin/env bash
# E-SELF-08b：v1 vs mma2 的机制读数 —— 达成带宽、占用率限制资源、warp 延迟首因。
# 判决：mma2 在更低占用率下若带宽更高 ⇒ 卡点是「在飞字节数」，木桶分析成立。
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
M="${M},launch__occupancy_limit_warps,launch__registers_per_thread"
M="${M},launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic"
M="${M},l1tex__t_sectors_pipe_lsu_mem_global_op_ldgsts_cache_access.sum"
M="${M},smsp__inst_executed.sum,smsp__inst_executed_op_global_ld.sum"
M="${M},smsp__inst_executed_op_shared_st.sum,smsp__inst_executed_op_ldsm.sum"
for r in barrier wait long_scoreboard short_scoreboard mio_throttle lg_throttle math_pipe_throttle not_selected selected drain; do
  M="${M},smsp__average_warp_latency_issue_stalled_${r}"
done

for A in v1 mma2; do
  case "$A" in
    v1)   KN="regex:grouped_mma_kernel" ;;
    mma2) KN="regex:grouped_mma2_kernel" ;;
  esac
  echo "=== $A ==="
  ncu --clock-control none --kernel-name "$KN" --launch-skip 3 --launch-count 1 \
      --metrics "$M" --csv --log-file "m2-${A}.csv" \
      python3 probe-mma2.py --n 2304 --only "$A" --ncu-warm > "m2-${A}.log" 2>&1 || {
        echo "NCU_FAILED $A"; tail -20 "m2-${A}.log"; exit 1; }
  test -s "m2-${A}.csv" || { echo "EMPTY_CSV $A"; exit 1; }
  echo "lines: $(wc -l < "m2-${A}.csv")"
done
echo M2NCU_OK