#!/usr/bin/env bash
# E-ROWS-02：访问粒度判决。
# 现象：warp 33.20%→49.78%（+49.94%）但有效带宽 150.13→148.44 GB/s（−1.13%）。
# 假设 H：~150 GB/s 是**访问粒度/sector 效率**造成的结构性上限，不是 MLP 不足。
#   若 DRAM 侧 sectors/request 明显 < 4（128B），则每次请求未填满 burst ⇒ 有效带宽被粒度腰斩，
#   此时增并发无效，必须改访问布局。
# 分离两侧：A=X 走 cp.async(LDGSTS)，B=权重 trellis words 走普通 global ld。
set -uo pipefail
cd /root/ptxwork

M="gpc__cycles_elapsed.avg,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ldgsts_cache_access.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ldgsts_cache_access.sum,\
lts__t_requests_op_read.sum,\
lts__t_sectors_op_read.sum,\
lts__t_requests_op_read_lookup_miss.sum,\
lts__t_sectors_op_read_lookup_miss.sum"

for arm in "$@"; do
  echo "=== 粒度 $arm ==="
  ncu --csv --log-file "gran-${arm}.csv" --metrics "$M" \
      -k "regex:grouped_mma_kernel" -c 1 -s 3 \
      python3 probe-rows.py --only "$arm" 2>&1 | tail -2
done
echo GRAN_DONE