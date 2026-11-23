#!/usr/bin/env bash
# 实测两节点 RoCE 可达带宽（单 rail），用于把「线速假设」换成实测分母。
# ib_write_bw 缓冲区仅数十 MB，引擎空载时运行；带内存闸。
# 用法：bash roce_bw.sh <label> [gid_index]
set -euo pipefail
LABEL="${1:?label}"
GID="${2:-5}"
WORKER=ztxjzh@192.168.100.11
DEV=rocep1s0f1
OUT=/tmp/roce-bw-$LABEL
MIN_GIB=5.0
mkdir -p "$OUT"

H=$(awk '/^MemAvailable:/{printf "%.3f", $2/1048576}' /proc/meminfo)
W=$(ssh -o BatchMode=yes "$WORKER" "awk '/^MemAvailable:/{printf \"%.3f\", \$2/1048576}' /proc/meminfo")
echo "[$LABEL] avail head=$H worker=$W GiB"
awk -v a="$H" -v b="$MIN_GIB" 'BEGIN{exit !(a<b)}' && { echo "ABORT head 内存"; exit 1; }
awk -v a="$W" -v b="$MIN_GIB" 'BEGIN{exit !(a<b)}' && { echo "ABORT worker 内存"; exit 1; }

{
  echo "label=$LABEL gid=$GID dev=$DEV"
  echo "head_mtu=$(cat /sys/class/net/enp1s0f1np1/mtu)"
  echo "worker_mtu=$(ssh -o BatchMode=yes "$WORKER" cat /sys/class/net/enp1s0f1np1/mtu)"
  echo "head_active_mtu=$(ibv_devinfo -d $DEV 2>/dev/null | awk '/active_mtu/{print $2}')"
} | tee "$OUT/config.txt"

for SZ in 65536 1048576 4194304 20971520; do
  echo "--- size $SZ ---"
  ssh -o BatchMode=yes "$WORKER" \
    "nohup timeout 60 ib_write_bw -d $DEV -x $GID -s $SZ -n 2000 --report_gbits \
     > /tmp/roce-bw-$LABEL-server-$SZ.txt 2>&1 &" >/dev/null 2>&1
  sleep 2
  timeout 60 ib_write_bw -d "$DEV" -x "$GID" -s "$SZ" -n 2000 --report_gbits \
    192.168.100.11 > "$OUT/client-$SZ.txt" 2>&1 || echo "  (client rc=$?)"
  # BW average 行
  awk -v s="$SZ" '/^[ ]*[0-9]/{printf "  size=%s iters=%s BW_peak=%s BW_avg=%s MsgRate=%s\n", $1,$2,$3,$4,$5}' \
    "$OUT/client-$SZ.txt" | tail -2
  sleep 1
done
echo "[$LABEL] 完成 -> $OUT"