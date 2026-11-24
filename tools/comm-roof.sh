#!/usr/bin/env bash
# 标定与生产同尺寸 AllGather 的可达上界：2048 行 x 5120 x fp32 = 41,943,040 B。
# 用镜像自带 tf-dsv41-comm（两 rank，CUDA event 计时，自带 gather_ok 正确性校验）。
# 显存约 126 MB，可与生产并存；生产此刻空闲，GPU 竞争可忽略。
set -u
B=/opt/tensorfold/bin/tf-dsv41-comm
IP=192.168.100.10
PORT=${PORT:-29777}
C=dsv41-1017-exl3-tf
O=/home/ztxjzh/ptxwork
EXTRA=${EXTRA:-}

docker exec $C bash -c "$EXTRA $B 0 $IP $PORT" > "$O/comm-roof-r0.json" 2> "$O/comm-roof-r0.err" &
P0=$!
sleep 3
ssh -o ConnectTimeout=8 ztxjzh@192.168.100.11 \
  "docker exec $C bash -c '$EXTRA $B 1 $IP $PORT'" > "$O/comm-roof-r1.json" 2> "$O/comm-roof-r1.err"
RC1=$?
wait $P0; RC0=$?
echo "rc0=$RC0 rc1=$RC1"
echo "--- rank0 ---"; cat "$O/comm-roof-r0.json"
echo "--- rank1 ---"; cat "$O/comm-roof-r1.json"
echo "--- stderr tail ---"; tail -4 "$O/comm-roof-r0.err"; tail -4 "$O/comm-roof-r1.err"