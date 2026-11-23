#!/usr/bin/env bash
# E-DECODE-01：decode 期设备级 GPU PM 采样。
# 复用 E-DRAM-01 r2 的已验证安全路径：独立容器 --memory=2g 硬上限、
# 无内核重放、不注入引擎、不改配置、不重启。
set -euo pipefail
IMG=tensorfold-dsv41:exl3-tp2-8348338a8d90
NAME=pm-decode
DUR=${1:-30}
FREQ=${2:-5000}
OUT=${3:-/tmp/pm-decode}
avail() { awk '/MemAvailable/{printf "%.2f", $2/1048576}' /proc/meminfo; }

A=$(avail)
echo "before avail_GiB: $A"
awk '/MemAvailable/{if ($2/1048576 < 5.0) {print "REFUSE: avail < 5 GiB"; exit 1}}' /proc/meminfo

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --memory=2g --memory-swap=2g --cap-add SYS_ADMIN \
  -v /tmp:/tmp "$IMG" \
  nsys profile --gpu-metrics-devices=all --gpu-metrics-set=gb20y-top \
  --gpu-metrics-frequency="$FREQ" --duration="$DUR" \
  --trace=none --sample=none --cpuctxsw=none -o "$OUT" --force-overwrite=true >/dev/null
sleep 3
echo "decode workload start $(date +%s.%N)"
python3 /tmp/decode_conc.py 4 700 0.0 1 2>&1 | tail -2
echo "decode workload end $(date +%s.%N)"
docker wait "$NAME" >/dev/null
docker rm "$NAME" >/dev/null
echo "after avail_GiB: $(avail)"
ls -la "${OUT}.nsys-rep"