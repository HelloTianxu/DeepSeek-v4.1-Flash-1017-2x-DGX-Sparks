#!/usr/bin/env bash
# 孤立通信 A/B：只改 bench 进程的 NCCL env，绝不影响生产容器。
# 每臂约 25s，无需重启服务。判据 us_2048_rows（= 生产同尺寸 41,943,040 B）。
set -u
O=/home/ztxjzh/ptxwork
: > "$O/comm-ab.txt"
run() {
  local name="$1" extra="$2"
  PORT=$((29800 + RANDOM % 100)) EXTRA="$extra" bash "$O/comm-roof.sh" > "$O/comm-ab-$name.raw" 2>&1
  local j0 j1
  j0=$(grep -o '{.*}' "$O/comm-ab-$name.raw" | head -1)
  j1=$(grep -o '{.*}' "$O/comm-ab-$name.raw" | sed -n 2p)
  printf '%-26s r0=%s\n' "$name" "${j0:-FAILED}" | tee -a "$O/comm-ab.txt"
  printf '%-26s r1=%s\n' "" "${j1:-FAILED}" >> "$O/comm-ab.txt"
}
run baseline       ""
run proto-simple   "NCCL_PROTO=Simple"
run qps2           "NCCL_IB_QPS_PER_CONNECTION=2"
run qps4           "NCCL_IB_QPS_PER_CONNECTION=4"
run chan32         "NCCL_MAX_NCHANNELS=32"
run chan16         "NCCL_MAX_NCHANNELS=16"
run buff8m         "NCCL_BUFFSIZE=8388608"
echo "=== done ==="
cat "$O/comm-ab.txt"