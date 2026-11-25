#!/usr/bin/env bash
# 收尾恢复核验：同工具同形状复测 prefill / N=1 / N=4，禁用 bench.py 跨尺寸中位。
# 口径来自 runs/forge-20261125-restore-verify/REPORT.md §2。
set -uo pipefail
cd "$HOME/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks"
OUT="$HOME/ptxwork/final-verify"
mkdir -p "$OUT"

echo "===== prefill x3 (cold_prompt.py 28038) ====="
for i in 1 2 3; do
  python3 tools/probes/cold_prompt.py "$OUT/pf-$i.json" 28038 >/dev/null 2>&1
  python3 - "$OUT/pf-$i.json" "$i" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
v=r.get("server_tok_s_under_ncu_NOT_A_SPEED_CLAIM")
print(f"  run{sys.argv[2]}: {v:.2f} tok/s  uncached={r.get('prompt_tokens_uncached','?')} pf_sec={r.get('prefill_seconds_delta',0):.4f}")
PY
  sleep 3
done

echo "===== N=1 decode (decode_probe.py 256 0.0 3) ====="
python3 tools/probes/decode_probe.py final-n1 256 0.0 3 2>&1 | tail -6

echo "===== N=4 aggregate (decode_conc.py 4 256 0.0 3) ====="
python3 tools/probes/decode_conc.py 4 256 0.0 3 2>&1 | tail -6

echo "===== prefill median ====="
python3 - "$OUT" <<'PY'
import json,statistics,pathlib,sys
d=pathlib.Path(sys.argv[1]); vs=[]
for i in (1,2,3):
    p=d/f"pf-{i}.json"
    if p.exists():
        v=json.load(open(p)).get("server_tok_s_under_ncu_NOT_A_SPEED_CLAIM")
        if v: vs.append(v)
if vs:
    m=statistics.median(vs)
    print(f"  values={[round(x,2) for x in vs]}")
    print(f"  MEDIAN={m:.2f} tok/s  baseline=2084.42  drift={(m/2084.42-1)*100:+.2f}%")
PY