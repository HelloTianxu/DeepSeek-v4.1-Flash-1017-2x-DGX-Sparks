#!/usr/bin/env bash
# E-SELF-11 前置检查：MODE 0 / 8 / 9 的机制读数 + 臂身份断言 + ldsm 保真断言。
# MODE 8 只摘 MMA，必须保住 ldmatrix.x4（ldsm 与 MODE 0 相等），否则它和 MODE 3 一样被编译器降级、读数作废。
set -euo pipefail
cd "$(dirname "$0")"

M="gpu__time_duration.sum"
M="${M},smsp__inst_executed.sum,smsp__inst_executed_op_ldsm.sum"
M="${M},smsp__inst_executed_op_shared_ld.sum,smsp__inst_executed_op_global_ld.sum"
M="${M},l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
M="${M},lts__t_sectors_op_read.sum"
M="${M},l1tex__throughput.avg.pct_of_peak_sustained_elapsed"
M="${M},sm__warps_active.avg.pct_of_peak_sustained_active"
M="${M},sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
M="${M},launch__registers_per_thread,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem"
M="${M},launch__occupancy_per_register_count,launch__occupancy_per_shared_mem_count"
for r in barrier wait long_scoreboard short_scoreboard mio_throttle lg_throttle \
         math_pipe_throttle not_selected selected dispatch_stall no_instruction; do
  M="${M},smsp__average_warp_latency_issue_stalled_${r}.pct"
done

for MODE in 0 10; do
  echo "=== MODE $MODE ==="
  ncu --clock-control none --kernel-name regex:grouped_mma2_kernel --launch-skip 3 --launch-count 1 \
      --metrics "$M" --csv --log-file "dc-m${MODE}.csv" \
      python3 probe-m2-parts.py --mode "$MODE" --ncu-warm --no-check > "dc-m${MODE}.log" 2>&1 || {
        echo "NCU_FAILED $MODE"; tail -20 "dc-m${MODE}.log"; exit 1; }
  test -s "dc-m${MODE}.csv" || { echo "EMPTY_CSV $MODE"; exit 1; }
  # 不用 `grep|head`：head 提前关管道会给 grep SIGPIPE，pipefail+set -e 会静默掐断整个循环
  NAMES=$(grep -o 'grouped_mma2_kernel<[^>]*>' "dc-m${MODE}.csv" | sort -u | tr '\n' ' ')
  test -n "$NAMES" || { echo "NO_NAME $MODE"; exit 1; }
  echo "  identity: $NAMES"
  echo "$NAMES" | grep -q "64, 0, 1, ${MODE}, 4" || { echo "IDENTITY_MISMATCH $MODE"; exit 1; }
done
echo PARTSNCU_OK