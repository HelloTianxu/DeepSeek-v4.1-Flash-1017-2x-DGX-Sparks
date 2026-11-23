#!/usr/bin/env bash
# 实测「推理期间」的内存最低点：回答 chunk 变大后还剩不剩地方跑推理。
# 只发一个冷前缀请求（服务的正常功能），全程 memguard 在位，不注入引擎、不 replay。
# 用法: ./measure_runtime_dip.sh [target_chars]
set -uo pipefail
CHARS="${1:-28000}"
WORKER=ztxjzh@192.168.100.11
OUT=/tmp/rt-dip
mkdir -p "$OUT"

avail() { awk '/^MemAvailable:/{printf "%.3f", $2/1048576}' /proc/meminfo; }

echo "=== 起点 ==="
H0=$(avail); W0=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$WORKER" \
  'awk "/^MemAvailable:/{printf \"%.3f\", \$2/1048576}" /proc/meminfo' 2>/dev/null || echo "?")
echo "head avail ${H0} GiB / worker avail ${W0} GiB"
awk -v a="$H0" 'BEGIN{if(a<5.0){print "ABORT: head avail < 5 GiB, 不做本次测量"; exit 1}}' || exit 1

# 采样器：200ms 一次，head 本地 + worker 远端
( while :; do printf '%s %s\n' "$(date +%s.%N)" "$(avail)"; sleep 0.2; done ) > "$OUT/head.txt" &
HP=$!
ssh -o BatchMode=yes "$WORKER" \
  'while :; do printf "%s %s\n" "$(date +%s.%N)" "$(awk "/^MemAvailable:/{printf \"%.3f\", \$2/1048576}" /proc/meminfo)"; sleep 0.2; done' \
  > "$OUT/worker.txt" 2>/dev/null &
WP=$!
sleep 2

echo "=== 发冷前缀请求 (target_chars=$CHARS) ==="
python3 /tmp/cold_prompt.py "$OUT/cold.json" "$CHARS" 2>&1 | tail -6
RC=$?

sleep 3
kill "$HP" "$WP" 2>/dev/null
wait 2>/dev/null

echo "=== 结果 ==="
for n in head worker; do
  awk -v n="$n" 'NF==2{v=$2; if(min==""||v<min)min=v; if(v>max)max=v; s+=v; c++}
    END{if(c)printf "%-7s 采样%d  最低 %.3f  最高 %.3f  均值 %.3f GiB\n", n, c, min, max, s/c}' "$OUT/$n.txt"
done
python3 - <<'PY'
import json,pathlib
p=pathlib.Path('/tmp/rt-dip/cold.json')
if p.exists():
    d=json.loads(p.read_text())
    for k in ('prompt_tokens','cached_tokens','prefill_seconds','server_tok_s'):
        if k in d: print(f"  {k} = {d[k]}")
PY
echo "=== 终点 ==="
echo "head avail $(avail) GiB"