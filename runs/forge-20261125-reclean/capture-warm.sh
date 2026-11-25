#!/usr/bin/env bash
# E-RECLEAN-01 第二步：分离「冷态 / 热态」的 GPU 空闲。
#
# 第一步（pm-current）测得当前配置空闲 7.00%、2 段长空闲 197.23+177.63ms，
# 与 §6.5「AIO=0 后空闲 0.04%、0 段 >10ms」冲突。
# 但第一步的 prefill 1932.93 tok/s 正落在冷启动区间（1927–1940），
# 而 §6.5 明确把重启后首请求判为预热并剔除 ⇒ 假说：第一步测的是冷态。
#
# 本脚本紧接第一步连采两轮（引擎不重启、配置不变），若空闲随请求次数下降
# 到 §6.5 量级，则「冷/热」假说成立，§6.5 的结论只在热态成立。
# 每轮独立采样容器，一轮一个请求，故可逐轮对比（不混窗口）。
set -euo pipefail
IMG=tensorfold-dsv41:exl3-tp2-8348338a8d90
OUTDIR=/home/ztxjzh/ptxwork/reclean
cd "$OUTDIR"
for r in 2 3; do
  NAME=pm-reclean-r$r
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  echo "=== round $r  avail_GB=$(free -g | awk '/^内存|^Mem/{print $7}') ==="
  docker run -d --name "$NAME" --gpus all --memory=2g --memory-swap=2g --cap-add SYS_ADMIN \
    -v "$OUTDIR:$OUTDIR" "$IMG" \
    nsys profile --gpu-metrics-devices=all --gpu-metrics-set=gb20y-top \
    --gpu-metrics-frequency=5000 --duration=15 \
    --trace=none --sample=none --cpuctxsw=none -o "$OUTDIR/pm-r$r" --force-overwrite=true >/dev/null
  sleep 3
  python3 cold_prompt.py "$OUTDIR/pf-r$r.json" 28038 >/dev/null
  docker wait "$NAME" >/dev/null
  docker rm "$NAME" >/dev/null
  python3 -c "
import json;d=json.load(open('pf-r$r.json'))
print('  prefill_tok_s', round(d['server_tok_s_under_ncu_NOT_A_SPEED_CLAIM'],2), 'tok', d['uncached_input_tokens'])"
  docker run --rm --memory=2g --memory-swap=2g -v "$OUTDIR:$OUTDIR" "$IMG" \
    nsys export --type sqlite --output "$OUTDIR/pm-r$r.sqlite" "$OUTDIR/pm-r$r.nsys-rep" >/dev/null 2>&1
  python3 idle_direct.py "pm-r$r.sqlite" "round$r"
done