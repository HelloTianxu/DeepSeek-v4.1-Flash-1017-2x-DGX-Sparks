#!/usr/bin/env bash
# E-ENGRAM-02: ENGRAM_AIO 对 decode 的独立判决（A/B/A 同场）
#
# 背景：AIO=0 在 11-24 fd1f811 入库，prefill +19.5%，但当时只做端到端否决，
# 从未单独测 decode。CURRENT-BASELINE:865 已注明「o.engram_aio 一个开关同管 decode+prompt」。
# 本实验补上缺失的那一半：decode 侧代价。
#
# 纪律：trap 强制恢复 local.sh 并重启；干预生效必须验到容器环境变量本身（§7.20）；
# 绝不并行两引擎，全部走 user systemd。
set -uo pipefail
cd "$HOME/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks" || exit 1
OUT="$HOME/ptxwork/engram-ab"
mkdir -p "$OUT"
KEY=$(cat "$HOME/.local/state/dsv41-tensorfold/api-key")
BK="$OUT/local.sh.bak"
cp scripts/local.sh "$BK"
echo "backup -> $BK  sha=$(sha256sum scripts/local.sh | cut -c1-16)"

wait_health() {
  for i in $(seq 1 90); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" http://127.0.0.1:8888/health 2>/dev/null || echo 000)
    if [ "$code" = "200" ]; then echo "  health 200 after ${i}s"; return 0; fi
    sleep 1
  done
  echo "  HEALTH TIMEOUT"; return 1
}

restore() {
  echo "=== TRAP 恢复 local.sh 并重启 ==="
  cp "$BK" scripts/local.sh
  systemctl --user restart dsv41-tensorfold.service
  wait_health
  echo "  restored TF_DS_ENGRAM_AIO=$(docker inspect dsv41-1017-exl3-tf --format '{{range .Config.Env}}{{println .}}{{end}}' | grep '^TF_DS_ENGRAM_AIO=' | cut -d= -f2)"
  echo "  sha=$(sha256sum scripts/local.sh | cut -c1-16)"
}
trap restore EXIT

measure() {
  local tag=$1
  echo "--- measure $tag : decode thinking ON ---"
  python3 "$HOME/ptxwork/decode_kind.py" 256 2 on  2>&1 | tee "$OUT/$tag-on.log"  | grep -vE '^SUMMARY'
  echo "--- measure $tag : decode thinking OFF ---"
  python3 "$HOME/ptxwork/decode_kind.py" 256 2 off 2>&1 | tee "$OUT/$tag-off.log" | grep -vE '^SUMMARY'
  echo "--- measure $tag : prefill x3 ---"
  for i in 1 2 3; do
    python3 tools/probes/cold_prompt.py "$OUT/$tag-pf-$i.json" 28038 >/dev/null 2>&1
    python3 -c "
import json,sys
r=json.load(open('$OUT/$tag-pf-$i.json'))
print('  pf$i %.2f tok/s'%r['server_tok_s_under_ncu_NOT_A_SPEED_CLAIM'])"
    sleep 2
  done
}

echo "===== 段 A0：当前生产 AIO=0 ====="
echo "  实测 TF_DS_ENGRAM_AIO=$(docker inspect dsv41-1017-exl3-tf --format '{{range .Config.Env}}{{println .}}{{end}}' | grep '^TF_DS_ENGRAM_AIO=' | cut -d= -f2)"
measure a0

echo "===== 切 AIO=1 并重启 ====="
printf '\n# E-ENGRAM-02 temporary A/B arm, removed by trap\nENGRAM_AIO=1\nexport TF_DS_ENGRAM_AIO=1\n' >> scripts/local.sh
systemctl --user restart dsv41-tensorfold.service
wait_health || exit 1
AIO=$(docker inspect dsv41-1017-exl3-tf --format '{{range .Config.Env}}{{println .}}{{end}}' | grep '^TF_DS_ENGRAM_AIO=' | cut -d= -f2)
echo "  干预生效核验 TF_DS_ENGRAM_AIO=$AIO"
if [ "$AIO" != "1" ]; then echo "  ABORT: 干预未生效，实测仍为 $AIO"; exit 1; fi

echo "===== 段 B：AIO=1 ====="
measure b1

echo "===== 段 A2 由 trap 恢复后另行复测 ====="