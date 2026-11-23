#!/usr/bin/env bash
# 两节点 CX7 直连口 MTU 变更（运行时，不持久化；重启即回原值）。
# 密码从 stdin 读入，脚本与产物中不含任何凭据。
# 用法：printf '%s\n' "$PW" | bash mtu_change.sh <mtu>
set -euo pipefail
MTU="${1:?target mtu}"
WORKER=ztxjzh@192.168.100.11
IFACES="enp1s0f1np1 enP2p1s0f1np1"
IFS= read -r PW

echo "=== 变更前 ==="
for i in $IFACES; do echo "head $i mtu=$(cat /sys/class/net/$i/mtu)"; done
ssh -o BatchMode=yes "$WORKER" "for i in $IFACES; do echo \"worker \$i mtu=\$(cat /sys/class/net/\$i/mtu)\"; done"

echo "=== 停服（避免 NCCL 连接中途断开）==="
systemctl --user stop dsv41-tensorfold.service
sleep 3
systemctl --user is-active dsv41-tensorfold.service || true
awk '/^MemAvailable:/{printf "停服后 head avail %.2f GiB\n", $2/1048576}' /proc/meminfo

echo "=== 改 MTU -> $MTU ==="
for i in $IFACES; do
  printf '%s\n' "$PW" | sudo -S -p '' ip link set dev "$i" mtu "$MTU" 2>/dev/null
done
printf '%s\n' "$PW" | ssh -o BatchMode=yes "$WORKER" \
  "IFS= read -r P; for i in $IFACES; do printf '%s\n' \"\$P\" | sudo -S -p '' ip link set dev \$i mtu $MTU 2>/dev/null; done"

sleep 2
echo "=== 变更后 ==="
for i in $IFACES; do echo "head $i mtu=$(cat /sys/class/net/$i/mtu)"; done
ssh -o BatchMode=yes "$WORKER" "for i in $IFACES; do echo \"worker \$i mtu=\$(cat /sys/class/net/\$i/mtu)\"; done"
for d in rocep1s0f1 roceP2p1s0f1; do
  echo "head $d state=$(cat /sys/class/infiniband/$d/ports/1/state) active_mtu=$(ibv_devinfo -d $d 2>/dev/null | awk '/active_mtu/{print $2}')"
done

echo "=== 连通性 ==="
ping -c 2 -W 2 192.168.100.11 >/dev/null 2>&1 && echo "rail1 OK" || echo "rail1 FAIL"
ping -c 2 -W 2 192.168.101.11 >/dev/null 2>&1 && echo "rail2 OK" || echo "rail2 FAIL"