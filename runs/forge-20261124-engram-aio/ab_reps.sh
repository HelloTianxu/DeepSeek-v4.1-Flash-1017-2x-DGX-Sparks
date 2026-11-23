#!/usr/bin/env bash
# Engram AIO A/B：N 次冷前缀重复，记录 tok/s 与内存最低点。
# 用法: ./ab_reps.sh <label> <reps> [target_chars]
set -uo pipefail
LABEL="${1:?label}"; REPS="${2:-2}"; CHARS="${3:-28000}"
OUT=/tmp/engram-ab; mkdir -p "$OUT"
avail() { awk '/^MemAvailable:/{printf "%.3f", $2/1048576}' /proc/meminfo; }

A0=$(avail)
echo "[$LABEL] 起点 head avail ${A0} GiB"
awk -v a="$A0" 'BEGIN{if(a<5.0){print "ABORT: avail<5GiB"; exit 1}}' || exit 1

( while :; do printf '%s %s\n' "$(date +%s.%N)" "$(avail)"; sleep 0.3; done ) > "$OUT/mem-$LABEL.txt" &
MP=$!

for i in $(seq 1 "$REPS"); do
  python3 /tmp/cold_prompt.py "$OUT/$LABEL-r$i.json" "$CHARS" >/dev/null 2>&1
  python3 - "$OUT/$LABEL-r$i.json" "$LABEL" "$i" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
tok=d.get('uncached_input_tokens'); sec=d.get('prefill_seconds_delta')
cached=d.get('prompt_tokens_cached_delta')
sp=tok/sec if tok and sec else 0
print(f"  {sys.argv[2]} r{sys.argv[3]}: tokens={tok:.0f} cached={cached:.0f} prefill={sec:.6f}s -> {sp:.2f} tok/s")
PY
  sleep 2
done

kill "$MP" 2>/dev/null; wait 2>/dev/null
awk 'NF==2{v=$2; if(m==""||v<m)m=v} END{printf "  内存最低 %.3f GiB\n", m}' "$OUT/mem-$LABEL.txt"
echo "[$LABEL] 终点 head avail $(avail) GiB"