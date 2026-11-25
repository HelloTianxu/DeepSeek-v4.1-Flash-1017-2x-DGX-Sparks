#!/usr/bin/env bash
# E-RECLEAN-01 第一步：零注入复验「当前配置」的 GPU 空闲占比。
#
# 为什么要有这一步：E-FEED-01/02 误用了 11-23（ENGRAM_AIO=1 时代）的 trace。
# §6.5 已判决 AIO=0 后空闲 0.04%，但那是 11-24 测的，之后又落地了 patch 0068（alias）。
# 本脚本在 alias 落地后的当前配置上复验，且**零停服**：引擎照常由 systemd 运行，
# PM 采样是设备级的，采样容器与引擎进程无关（沿用 E-DRAM-01 r2 / §6.5 同法）。
#
# 硬约束 §0：采样容器带 --memory=2g --memory-swap=2g 硬上限，OOM 只杀本容器。
# 用法：bash capture-current.sh [duration_s] [freq_hz] [out_prefix]
set -euo pipefail
IMG=tensorfold-dsv41:exl3-tp2-8348338a8d90
NAME=pm-reclean-01
DUR=${1:-20}
FREQ=${2:-5000}
OUTDIR=${3:-/home/ztxjzh/ptxwork/reclean}
OUT="$OUTDIR/pm-current"
avail() { free -g | awk '/^内存|^Mem/{print $7}'; }

mkdir -p "$OUTDIR"
docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "before avail_GB: $(avail)"
echo "engine image: $(docker inspect --format '{{.Config.Image}}' dsv41-1017-exl3-tf)"
echo "engine AIO:   $(docker exec dsv41-1017-exl3-tf printenv TF_DS_ENGRAM_AIO)"
echo "engine alias: $(docker exec dsv41-1017-exl3-tf printenv TF_DS_PROMPT_ALIAS)"

docker run -d --name "$NAME" --gpus all --memory=2g --memory-swap=2g --cap-add SYS_ADMIN \
  -v "$OUTDIR:$OUTDIR" "$IMG" \
  nsys profile --gpu-metrics-devices=all --gpu-metrics-set=gb20y-top \
  --gpu-metrics-frequency="$FREQ" --duration="$DUR" \
  --trace=none --sample=none --cpuctxsw=none -o "$OUT" --force-overwrite=true >/dev/null
sleep 3
# 权威口径：28038 字符 / 10752 token 冷前缀，nonce 置位置 0，max_tokens=1
python3 "$OUTDIR/cold_prompt.py" "$OUTDIR/pf-current.json" 28038
docker wait "$NAME" >/dev/null
docker rm "$NAME" >/dev/null
echo "after avail_GB: $(avail)"
ls -la "${OUT}.nsys-rep"