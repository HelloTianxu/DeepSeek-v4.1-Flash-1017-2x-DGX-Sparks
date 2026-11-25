#!/usr/bin/env bash
# E-FEED-02 A/B：同一二进制 pinned-ab，只翻 TF_DS_ENGRAM_PINNED。
#
# 纪律：对照臂必须同场重测（不复用历史基线数值）；A/B/A/B/A/B 交替看配对差是否同向。
# 口径：cold_prompt.py 28038 字符（10752 token 冷前缀，nonce 置位置 0），
#       主报 uncached_input_tokens / prefill_seconds_delta；decode 用 decode_probe.py 同形状。
# 身份断言：PINNED=1 必须在引擎日志里出现 "page-locked"，否则说明 env 没传到，该臂作废。
set -uo pipefail
PROJ=$HOME/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks
OUT=$HOME/ptxwork/feed-ab
IMG=tensorfold-dsv41:pinned-ab
ROUNDS=${ROUNDS:-3}
mkdir -p "$OUT"
cd "$PROJ"

[ -f scripts/local.sh.feed-bak ] || cp scripts/local.sh scripts/local.sh.feed-bak

restore() {
  cp scripts/local.sh.feed-bak scripts/local.sh
  echo "== local.sh 已还原为基线 =="
}
trap restore EXIT

set_arm() {
  python3 - "$1" "$IMG" <<'PY'
import sys, pathlib, re
v, img = sys.argv[1], sys.argv[2]
t = pathlib.Path("scripts/local.sh.feed-bak").read_text()
t2, n = re.subn(r'^IMAGE=.*$', 'IMAGE=' + img, t, flags=re.M)
assert n == 1, f"IMAGE line count {n}"
t2 = t2.rstrip() + "\nexport TF_DS_ENGRAM_PINNED=" + v + "\n"
pathlib.Path("scripts/local.sh").write_text(t2)
print(f"  arm PINNED={v} IMAGE={img}")
PY
}

wait_ready() {
  for i in $(seq 1 45); do
    sleep 3
    code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8888/health 2>/dev/null || echo 000)
    if [ "$code" = "200" ]; then
      av=$(free -g | awk 'NR==2{print $7}')
      if [ "$av" -le 12 ]; then echo "  ready t=$((i*3))s avail=${av}GiB"; return 0; fi
    fi
  done
  echo "  NOT_READY"; return 1
}

for round in $(seq 1 "$ROUNDS"); do
  for arm in 0 1; do
    tag="r${round}-p${arm}"
    echo "== $tag =="
    set_arm "$arm" || exit 1
    t0=$(date '+%Y-%m-%d %H:%M:%S')
    systemctl --user restart dsv41-tensorfold.service
    wait_ready || { echo "ABORT: engine not ready for $tag"; exit 1; }

    # 身份断言：只看本次 restart 之后的日志（固定窗口会覆盖上一臂，造成假断言）
    pl=$(journalctl --user -u dsv41-tensorfold.service --since "$t0" --no-pager 2>/dev/null | grep -c "page-locked" || true)
    echo "  page-locked log lines: $pl"
    if [ "$arm" = "1" ] && [ "$pl" -lt 1 ]; then
      echo "ABORT: PINNED=1 但无 page-locked 日志，env 未生效"; exit 1
    fi
    if [ "$arm" = "0" ] && [ "$pl" -ge 1 ]; then
      echo "ABORT: PINNED=0 却出现 page-locked 日志"; exit 1
    fi

    for r in 1 2 3; do
      python3 tools/probes/cold_prompt.py "$OUT/pf-$tag-$r.json" 28038 >/dev/null 2>&1
      python3 - "$OUT/pf-$tag-$r.json" "$tag" "$r" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
u, s = d["uncached_input_tokens"], d["prefill_seconds_delta"]
print(f"  {sys.argv[2]} run{sys.argv[3]}: tok={u} pf={s:.4f}s tok_s={u/s:.2f} cached={d['prompt_tokens_cached_delta']}")
PY
    done
    python3 tools/probes/decode_probe.py "$tag" 256 0.0 3 > "$OUT/dec-$tag.json" 2>/dev/null
    python3 - "$OUT/dec-$tag.json" "$tag" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
k = "decode_tok_s_median_excl_first"
print(f"  {sys.argv[2]} decode N1: {d.get(k)}")
PY
  done
done
echo AB_DONE