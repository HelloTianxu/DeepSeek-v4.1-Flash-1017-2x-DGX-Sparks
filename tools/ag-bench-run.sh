#!/usr/bin/env bash
# 跑 nccl-ag-bench 双 rank：扫 6 个行数 x {fp32, bf16}，与生产同 NCCL/IB/GDR 路径。
set -u
O=/home/ztxjzh/ptxwork
C=dsv41-1017-exl3-tf
IP=192.168.100.10
PORT=${PORT:-29955}
W=ztxjzh@192.168.100.11

ssh -o ConnectTimeout=8 "$W" "mkdir -p /home/ztxjzh/ptxwork" || exit 1
scp -q "$O/nccl-ag-bench" "$W:/home/ztxjzh/ptxwork/nccl-ag-bench" || exit 1
ssh -o ConnectTimeout=8 "$W" "docker cp /home/ztxjzh/ptxwork/nccl-ag-bench $C:/tmp/nccl-ag-bench" || exit 1

docker exec $C /tmp/nccl-ag-bench 0 "$IP" "$PORT" > "$O/ag-r0.json" 2> "$O/ag-r0.err" &
P0=$!
sleep 3
ssh -o ConnectTimeout=8 "$W" "docker exec $C /tmp/nccl-ag-bench 1 $IP $PORT" > "$O/ag-r1.json" 2> "$O/ag-r1.err"
RC1=$?
wait $P0; RC0=$?
echo "rc0=$RC0 rc1=$RC1"
echo "--- rank0 ---"; cat "$O/ag-r0.json"
echo "--- rank1 ---"; cat "$O/ag-r1.json"
echo "--- err ---"; tail -3 "$O/ag-r0.err"; tail -3 "$O/ag-r1.err"