#!/usr/bin/env bash
# E-SELF-07d：让 ncu 直接报占用率的限制资源，并确认权重暂存是否走寄存器中转（无 cp.async）。
set -euo pipefail
cd "$(dirname "$0")"

M="launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem"
M="${M},launch__occupancy_limit_blocks,launch__occupancy_limit_warps"
M="${M},launch__occupancy_per_register_count,launch__occupancy_per_shared_mem_count"
M="${M},launch__registers_per_thread,launch__shared_mem_per_block_static"
M="${M},launch__shared_mem_config_size,launch__waves_per_multiprocessor"
M="${M},l1tex__t_sectors_pipe_lsu_mem_global_op_ldgsts_cache_access.sum"
M="${M},l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"
M="${M},l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio"

for W in mma rows; do
  case "$W" in
    mma)  KN="regex:grouped_mma_kernel" ;;
    rows) KN="regex:grouped_rows_kernel" ;;
  esac
  echo "=== $W ==="
  ncu --clock-control none --kernel-name "$KN" --launch-skip 3 --launch-count 1 \
      --metrics "$M" --csv --log-file "occ-${W}.csv" \
      python3 probe-layer.py --which "$W" --ncu-warm > "occ-${W}.log" 2>&1 || {
        echo "NCU_FAILED $W"; tail -20 "occ-${W}.log"; exit 1; }
  test -s "occ-${W}.csv" || { echo "EMPTY_CSV $W"; exit 1; }
  python3 - "$W" <<'PY'
import csv, sys
w = sys.argv[1]
lines = open(f"occ-{w}.csv").read().splitlines()
s = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
for r in csv.DictReader(lines[s:]):
    print(f"  {r['Metric Name']:<66}{r['Metric Value']:>16}  {r['Metric Unit']}")
PY
done
echo OCC_OK