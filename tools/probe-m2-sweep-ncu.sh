#!/usr/bin/env bash
# E-SELF-09b：扫描臂的机制读数 + **臂身份断言**（防止环境变量没生效却把噪声当结果）。
set -euo pipefail
cd "$(dirname "$0")"

M="gpu__time_duration.sum"
M="${M},lts__t_sectors_op_read.sum,lts__t_sectors_op_read_lookup_miss.sum"
M="${M},lts__throughput.avg.pct_of_peak_sustained_elapsed"
M="${M},l1tex__throughput.avg.pct_of_peak_sustained_elapsed"
M="${M},sm__throughput.avg.pct_of_peak_sustained_elapsed"
M="${M},sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
M="${M},sm__warps_active.avg.pct_of_peak_sustained_active"
M="${M},launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem"
M="${M},launch__registers_per_thread,launch__shared_mem_per_block_dynamic"
M="${M},launch__occupancy_per_register_count,launch__occupancy_per_shared_mem_count"
M="${M},smsp__inst_executed.sum"
for r in barrier wait long_scoreboard short_scoreboard mio_throttle lg_throttle math_pipe_throttle not_selected selected; do
  M="${M},smsp__average_warp_latency_issue_stalled_${r}"
done

for A in 0:1 3:1 3:2 6:1; do
  TAG="${A//:/-}"
  echo "=== arm $A ==="
  ncu --clock-control none --kernel-name regex:grouped_mma2_kernel --launch-skip 3 --launch-count 1 \
      --metrics "$M" --csv --log-file "sw-${TAG}.csv" \
      python3 probe-m2-sweep.py --arm "$A" --ncu-warm --no-check > "sw-${TAG}.log" 2>&1 || {
        echo "NCU_FAILED $A"; tail -20 "sw-${TAG}.log"; exit 1; }
  test -s "sw-${TAG}.csv" || { echo "EMPTY_CSV $A"; exit 1; }
  # 臂身份断言：模板实参必须与请求的 S/MINB 一致
  S="${A%%:*}"; B="${A##*:}"
  if grep -q "grouped_mma2_kernel<2, 6, 6, 64, ${S}, ${B}>" "sw-${TAG}.csv"; then
    echo "  identity OK: <2,6,6,64,${S},${B}>"
  else
    echo "  IDENTITY_MISMATCH arm=$A"; grep -o "grouped_mma2_kernel<[^>]*>" "sw-${TAG}.csv" | head -2; exit 1
  fi
done
echo SWNCU_OK