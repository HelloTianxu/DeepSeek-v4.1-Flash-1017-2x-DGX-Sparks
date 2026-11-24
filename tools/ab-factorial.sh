#!/usr/bin/env bash
# E-SELF-04b §5.3: separate the two factors in ONE session.
#
# Along the down rows shape, two things move independently:
#   weight-side work (decode ALU + weight bytes) ~ MG = maxm/(16*G)      -> falls as G rises
#   activation-side X reads ~ column blocks N/(NT*16)                    -> rises as NT falls
# G=4,NT=4 moved both at once (decode /2, activations x2) and lost 1.84%, so the two must be
# separated before either can be called the binding constraint.
#
# Cells (G:NT), PF=1 and W=4 everywhere, both served k2 ranges present:
#   2:8  baseline    decode x1   activations x1   (G*NT=16, acc 128 regs)
#   1:8              decode x2   activations x1   (G*NT=8,  acc 64)
#   2:4              decode x1   activations x2   (G*NT=8,  acc 64)
#   4:4              decode /2   activations x2   (G*NT=16, acc 128)
# A fresh baseline is measured before and after each test cell, so every comparison is a
# same-session pair (handbook: never reuse a historical baseline).
set -eu
P=$HOME/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks
L=$P/scripts/local.sh
OUT=${1:?out log}
KEY=$(cat "$HOME/.local/state/dsv41-tensorfold/api-key")
CELLS=(2:8 1:8 2:8 2:4 2:8 4:4)

[ -f "$L.pre-gn-bak" ] || cp -p "$L" "$L.pre-gn-bak"
sed -i 's|^IMAGE=.*|IMAGE=tensorfold-dsv41:base-gn|' "$L"

setenv_line() { # name value
  if grep -q "^export $1=" "$L"; then
    sed -i "s|^export $1=.*|export $1=$2|" "$L"
  else
    printf '\nexport %s=%s\n' "$1" "$2" >> "$L"
  fi
}
ready() {
  for i in $(seq 1 60); do
    curl -s -m 3 -H "Authorization: Bearer $KEY" http://127.0.0.1:8888/health 2>/dev/null | grep -q '"ok"' && return 0
    sleep 3
  done
  return 1
}

echo "arm  G  NT   uncached_tok  prefill_s   tok_per_s" | tee "$OUT"
n=0
for cell in "${CELLS[@]}"; do
  n=$((n+1)); g=${cell%%:*}; nt=${cell##*:}
  setenv_line TF_DS_EXPERT_G "$g"
  setenv_line TF_DS_EXPERT_NT "$nt"
  systemctl --user stop dsv41-tensorfold.service >/dev/null 2>&1 || true
  sleep 8
  systemctl --user reset-failed dsv41-tensorfold.service >/dev/null 2>&1 || true
  systemctl --user start dsv41-tensorfold.service
  if ! ready; then
    echo "$n $g $nt FAILED_READY" | tee -a "$OUT"
    journalctl --user -u dsv41-tensorfold.service -n 12 --no-pager 2>/dev/null | tail -6 | tee -a "$OUT"
    continue
  fi
  for rep in 1 2; do
    f=/tmp/fact-$n-$rep.json
    python3 /tmp/cold_prompt.py "$f" 28000 >/dev/null 2>&1 || { echo "$n $g $nt PROBE_FAILED" | tee -a "$OUT"; continue; }
    python3 - "$f" "$g" "$nt" "$n" >> "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
t = d.get("uncached_input_tokens"); s = d.get("prefill_seconds_delta")
print(f"{sys.argv[4]:>3}  {sys.argv[2]:>2} {sys.argv[3]:>3}  {t:>12}  {s:>9.4f}  {t/s:>9.2f}"
      if t and s else f"{sys.argv[4]:>3}  {sys.argv[2]:>2} {sys.argv[3]:>3}  MISSING_FIELDS")
PY
  done
  tail -2 "$OUT"
done
echo FACTORIAL_DONE | tee -a "$OUT"