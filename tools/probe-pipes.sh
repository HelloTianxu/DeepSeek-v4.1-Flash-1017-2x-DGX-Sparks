#!/usr/bin/env bash
# E-SELF-07b：各执行管线饱和度 + 指令成分。
# 回答「蛋厂是不是已经满负荷」：若 ALU/FMA 管线接近 100%，则造币车间本身是卡口，
# 卡车（DRAM）空着是结果而非原因。
set -euo pipefail
cd "$(dirname "$0")"

P=""
for p in alu aluheavy fma fmaheavy fmalite fp64 tensor tensor_subpipe_hmma; do
  P="${P},sm__pipe_${p}_cycles_active.avg.pct_of_peak_sustained_active"
done
P="${P},sm__pipe_fmaheavy_subpipe_alulite_cycles_active.avg.pct_of_peak_sustained_active"
P="${P},sm__inst_issued.avg.pct_of_peak_sustained_active"
P="${P},smsp__issue_active.avg.pct_of_peak_sustained_active"
P="${P#,}"

I="smsp__inst_executed.sum"
for o in shared_ld shared_st ldsm global_ld global_st branch; do
  I="${I},smsp__inst_executed_op_${o}.sum"
done
I="${I},l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
I="${I},l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
I="${I},sm__warps_active.avg.pct_of_peak_sustained_active"
I="${I},gpu__time_duration.sum"

for W in mma rows; do
  case "$W" in
    mma)  KN="regex:grouped_mma_kernel" ;;
    rows) KN="regex:grouped_rows_kernel" ;;
  esac
  echo "=== $W ==="
  ncu --clock-control none --kernel-name "$KN" --launch-skip 3 --launch-count 1 \
      --metrics "${P},${I}" --csv --log-file "pipe-${W}.csv" \
      python3 probe-layer.py --which "$W" --ncu-warm > "pipe-${W}.log" 2>&1 || {
        echo "NCU_FAILED $W"; tail -20 "pipe-${W}.log"; exit 1; }
  test -s "pipe-${W}.csv" || { echo "EMPTY_CSV $W"; exit 1; }
  echo "lines: $(wc -l < "pipe-${W}.csv")"
done
echo PIPE_OK