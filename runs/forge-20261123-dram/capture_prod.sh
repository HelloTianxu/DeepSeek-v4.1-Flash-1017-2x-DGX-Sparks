#!/usr/bin/env bash
# E-DRAM-01 r2 生产采集：零注入、无内核重放、容器带硬内存上限。
# 引擎照常由 systemd 运行，本脚本不重启、不改配置、不注入引擎进程。
# GPU PM 采样是设备级的，采样容器与引擎进程无关。
set -euo pipefail
IMG=tensorfold-dsv41:exl3-tp2-8348338a8d90
NAME=pm-probe-r2
DUR=${1:-30}
FREQ=${2:-5000}
OUT=${3:-/tmp/pm-prod-r2}
avail() { free -g | awk '/^内存|^Mem/{print $7}'; }

docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "before avail_GB: $(avail)"
# 硬上限 2g：OOM 时杀本容器，不影响引擎与整机（硬约束 §0）
docker run -d --name "$NAME" --gpus all --memory=2g --memory-swap=2g --cap-add SYS_ADMIN \
  -v /tmp:/tmp "$IMG" \
  nsys profile --gpu-metrics-devices=all --gpu-metrics-set=gb20y-top \
  --gpu-metrics-frequency="$FREQ" --duration="$DUR" \
  --trace=none --sample=none --cpuctxsw=none -o "$OUT" --force-overwrite=true >/dev/null
sleep 3
python3 /tmp/cold_prompt.py /tmp/cold-r2.json
docker wait "$NAME" >/dev/null
docker rm "$NAME" >/dev/null
echo "after avail_GB: $(avail)"
ls -la "${OUT}.nsys-rep"