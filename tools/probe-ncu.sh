#!/usr/bin/env bash
# E-SELF-07：离线单内核 warp 延迟分解 + 各级供给吞吐。
# 从消费者 SM 往上游回溯：SM 发不出指令 → warp 卡在哪一环 → 那一环是否饱和。
# --clock-control none：保持用户实际时钟（已主动降频），与 E-DRAM-01 的校准口径一致。
set -euo pipefail
cd "$(dirname "$0")"

STALL=""
for r in selected not_selected wait short_scoreboard long_scoreboard barrier \
         mio_throttle lg_throttle math_pipe_throttle tex_throttle dispatch_stall \
         drain membar no_instruction branch_resolving misc sleeping; do
  STALL="${STALL},smsp__average_warp_latency_issue_stalled_${r}"
done
STALL="${STALL#,}"

THRU="sm__throughput.avg.pct_of_peak_sustained_elapsed"
THRU="${THRU},lts__throughput.avg.pct_of_peak_sustained_elapsed"
THRU="${THRU},l1tex__throughput.avg.pct_of_peak_sustained_elapsed"
THRU="${THRU},sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
THRU="${THRU},sm__warps_active.avg.pct_of_peak_sustained_active"
THRU="${THRU},sm__inst_issued.avg.per_cycle_active"
THRU="${THRU},lts__t_sectors_op_read.sum"
THRU="${THRU},lts__t_sectors_op_read_lookup_miss.sum"
THRU="${THRU},lts__t_sectors_op_write.sum"
THRU="${THRU},l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"
THRU="${THRU},l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
THRU="${THRU},smsp__inst_executed.sum"
THRU="${THRU},gpu__time_duration.sum"

for W in mma rows; do
  case "$W" in
    mma)  KN="regex:grouped_mma_kernel" ;;
    rows) KN="regex:grouped_rows_kernel" ;;
  esac
  echo "=== $W ==="
  ncu --clock-control none --kernel-name "$KN" --launch-skip 3 --launch-count 1 \
      --metrics "${STALL},${THRU}" --csv --log-file "ncu-${W}.csv" \
      python3 probe-layer.py --which "$W" --ncu-warm > "ncu-${W}.log" 2>&1 || {
        echo "NCU_FAILED $W"; tail -20 "ncu-${W}.log"; exit 1; }
  test -s "ncu-${W}.csv" || { echo "EMPTY_CSV $W"; exit 1; }
  echo "rows: $(wc -l < "ncu-${W}.csv")"
done
echo NCU_OK