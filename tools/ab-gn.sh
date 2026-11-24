#!/usr/bin/env bash
# E-SELF-04b same-session A/B/A/B: TF_DS_EXPERT_G=2 (control) vs 4 (test).
# One image, one cubin, one config; only the env differs, so the control arm is re-measured in the
# same session rather than compared against a historical number (handbook: never reuse a baseline).
#
# G*NT stays 16, so acc registers (128) and the program count are invariant; MG = ceil(maxm/(16*G))
# halves at G=4, halving total weight-tile decodes. The rising cost is 2G barriers per program.
#
# Usage: bash ab-gn.sh OUT_LOG [arms...]   default arms: 2 4 2 4
set -eu
P=$HOME/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks
L=$P/scripts/local.sh
OUT=${1:?out log}
shift || true
ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || ARMS=(2 4 2 4)
KEY=$(cat "$HOME/.local/state/dsv41-tensorfold/api-key")

[ -f "$L.pre-gn-bak" ] || cp -p "$L" "$L.pre-gn-bak"
sed -i 's|^IMAGE=.*|IMAGE=tensorfold-dsv41:base-gn|' "$L"

ready() {
  for i in $(seq 1 60); do
    if curl -s -m 3 -H "Authorization: Bearer $KEY" http://127.0.0.1:8888/health 2>/dev/null | grep -q '"ok"'; then
      return 0
    fi
    sleep 3
  done
  return 1
}

echo "arm  G   uncached_tok  prefill_s   tok_per_s" | tee "$OUT"
for g in "${ARMS[@]}"; do
  # rewrite the switch line in place (append once, then edit)
  if grep -q '^export TF_DS_EXPERT_G=' "$L"; then
    sed -i "s|^export TF_DS_EXPERT_G=.*|export TF_DS_EXPERT_G=$g|" "$L"
  else
    printf '\n# E-SELF-04b: down member tiles a program; NT=16/G keeps G*NT=16\nexport TF_DS_EXPERT_G=%s\n' "$g" >> "$L"
  fi
  systemctl --user stop dsv41-tensorfold.service >/dev/null 2>&1 || true
  sleep 8
  systemctl --user reset-failed dsv41-tensorfold.service >/dev/null 2>&1 || true
  systemctl --user start dsv41-tensorfold.service
  if ! ready; then
    echo "G=$g  FAILED_TO_BECOME_READY" | tee -a "$OUT"
    journalctl --user -u dsv41-tensorfold.service -n 15 --no-pager 2>/dev/null | tail -8 | tee -a "$OUT"
    continue
  fi
  # two cold-prefix probes an arm; the first settles caches, the second is the reading
  for rep in 1 2; do
    f=/tmp/gn-$g-$rep.json
    python3 /tmp/cold_prompt.py "$f" 28000 >/dev/null 2>&1 || { echo "G=$g rep$rep PROBE_FAILED" | tee -a "$OUT"; continue; }
    python3 - "$f" "$g" "$rep" >> "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
t = d.get("uncached_input_tokens"); s = d.get("prefill_seconds_delta")
if t and s:
    print(f"{sys.argv[3]:>3}  {sys.argv[2]:>2}  {t:>12}  {s:>9.4f}  {t/s:>9.2f}")
else:
    print(f"{sys.argv[3]:>3}  {sys.argv[2]:>2}  MISSING_FIELDS {list(d)[:6]}")
PY
  done
  tail -2 "$OUT"
done
echo AB_GN_DONE | tee -a "$OUT"