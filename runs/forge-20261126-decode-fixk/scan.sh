#!/usr/bin/env bash
# E-DECODE-03: confidence 卡口量化 —— TF_DS_FIX_K 扫描（单路 decode）
#
# 依据 E-DECODE-01:107-111：depth 上限按 live streams 算，单路 live=1 ⇒ 上限 16/1-1=15，
# 实测 tok/round 仅 2.05-4.74 ⇒ 「不是上限在卡，是 confidence 在卡」。
# FIX_K 强制每轮验证 K 个草稿，用来量化放开 confidence 后的收益或代价。
#
# 纪律：trap 强制恢复；干预生效须验容器环境变量；绝不并行两引擎。
set -uo pipefail
cd "$HOME/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks" || exit 1
OUT="$HOME/ptxwork/fixk"
mkdir -p "$OUT"
KEY=$(cat "$HOME/.local/state/dsv41-tensorfold/api-key")
BK="$OUT/local.sh.bak"
cp scripts/local.sh "$BK"
echo "backup sha=$(sha256sum scripts/local.sh | cut -c1-16)"

wait_health() {
  for i in $(seq 1 90); do
    c=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" http://127.0.0.1:8888/health 2>/dev/null || echo 000)
    [ "$c" = "200" ] && { echo "  health 200 after ${i}s"; return 0; }
    sleep 1
  done
  echo "  HEALTH TIMEOUT"; return 1
}
envval() { docker inspect dsv41-1017-exl3-tf --format '{{range .Config.Env}}{{println .}}{{end}}' | grep "^$1=" | cut -d= -f2; }

restore() {
  echo "=== TRAP 恢复 ==="
  cp "$BK" scripts/local.sh
  systemctl --user restart dsv41-tensorfold.service
  wait_health
  echo "  FIX_K=[$(envval TF_DS_FIX_K)]  AIO=$(envval TF_DS_ENGRAM_AIO)  sha=$(sha256sum scripts/local.sh | cut -c1-16)"
}
trap restore EXIT

measure() {
  local tag=$1
  echo "--- $tag : thinking ON ---"
  python3 "$HOME/ptxwork/decode_kind.py" 256 2 on  2>&1 | tee "$OUT/$tag-on.log"  | grep -vE '^SUMMARY'
  echo "--- $tag : thinking OFF ---"
  python3 "$HOME/ptxwork/decode_kind.py" 256 2 off 2>&1 | tee "$OUT/$tag-off.log" | grep -vE '^SUMMARY'
}

arm() {
  local k=$1
  cp "$BK" scripts/local.sh
  if [ "$k" != "unset" ]; then
    printf '\n# E-DECODE-03 temporary arm, removed by trap\nexport TF_DS_FIX_K=%s\n' "$k" >> scripts/local.sh
  fi
  systemctl --user restart dsv41-tensorfold.service
  wait_health || return 1
  local got=$(envval TF_DS_FIX_K)
  echo "  干预核验 TF_DS_FIX_K=[$got] (期望 $k)"
  if [ "$k" = "unset" ]; then
    [ -n "$got" ] && { echo "  ABORT: 期望未设但实为 $got"; return 1; }
  else
    [ "$got" = "$k" ] || { echo "  ABORT: 干预未生效"; return 1; }
  fi
}

echo "===== 臂 1: FIX_K unset（当前生产 = confidence 自适应） ====="
echo "  FIX_K=[$(envval TF_DS_FIX_K)]"
measure k-unset

for K in 4 8; do
  echo "===== 臂: FIX_K=$K ====="
  arm "$K" || exit 1
  measure "k$K"
done

echo "===== 完成，trap 恢复生产 ====="