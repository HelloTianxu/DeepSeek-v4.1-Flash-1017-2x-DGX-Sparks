#!/usr/bin/env bash
# Post-boot / post-deploy self-check. RUN THIS ON THE HEAD NODE.
#
# Answers one question: is this machine serving the best known baseline, as it should after
# every reboot? See docs/CURRENT-BASELINE.md §6.
#
#   tools/check-boot.sh
#
# Exit 0 = all checks pass. Non-zero = at least one failed (each failure is printed).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

IMAGE_WANT=${IMAGE_WANT:-tensorfold-dsv41:base-alias}
CONTAINER=${CONTAINER:-dsv41-1017-exl3-tf}
UNIT=dsv41-tensorfold.service
fails=0
pass() { printf '  ok    %s\n' "$1"; }
bad()  { printf '  FAIL  %s\n' "$1"; fails=$((fails + 1)); }

echo "== 配置口径 =="
if [[ ! -f scripts/local.sh ]]; then
  bad "scripts/local.sh missing (materialize from scripts/local.sh.authoritative)"
elif diff -q scripts/local.sh scripts/local.sh.authoritative >/dev/null 2>&1; then
  pass "local.sh == local.sh.authoritative"
else
  bad "local.sh differs from local.sh.authoritative -- run tools/deploy-baseline.sh"
  diff scripts/local.sh.authoritative scripts/local.sh | head -20 | sed 's/^/        /'
fi

echo "== 开机自启链路 =="
[[ "$(loginctl show-user "$USER" --property=Linger --value 2>/dev/null)" == yes ]] \
  && pass "Linger=yes" || bad "Linger is not yes (user service will not start at boot)"
[[ "$(systemctl --user is-enabled "$UNIT" 2>/dev/null)" == enabled ]] \
  && pass "unit enabled" || bad "unit not enabled"
systemctl --user show "$UNIT" --property=WantedBy --value 2>/dev/null | grep -q default.target \
  && pass "WantedBy=default.target" || bad "unit not wanted by default.target"
systemctl is-enabled docker >/dev/null 2>&1 && pass "docker enabled" || bad "docker not enabled"

echo "== 运行态 =="
KEY_FILE="$HOME/.local/state/dsv41-tensorfold/api-key"
if [[ -r "$KEY_FILE" ]]; then
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 8 http://127.0.0.1:8888/health \
         -H "Authorization: Bearer $(cat "$KEY_FILE")" 2>/dev/null)
  [[ "$code" == 200 ]] && pass "/health 200" || bad "/health returned '$code'"
else
  bad "api key file not readable: $KEY_FILE"
fi

if docker image inspect "$IMAGE_WANT" >/dev/null 2>&1; then
  pass "head image $IMAGE_WANT present"
else
  bad "head image $IMAGE_WANT missing"
fi

envdump=$(docker exec "$CONTAINER" env 2>/dev/null)
if [[ -z "$envdump" ]]; then
  bad "container $CONTAINER not running"
else
  grep -q '^TF_DS_PROMPT_ALIAS=1$' <<<"$envdump" \
    && pass "TF_DS_PROMPT_ALIAS=1 (patch 0068 active)" || bad "PROMPT_ALIAS not 1"
  grep -q '^TF_DS_L2_PREFETCH=1$' <<<"$envdump" \
    && pass "TF_DS_L2_PREFETCH=1 (worth +6.3% single stream)" || bad "L2_PREFETCH not 1"
  # None of the rejected experiment switches may be present.
  stray=$(grep -cE '^TF_DS_(NO_COPIES|SOLO_OVERHEAD|IDX_SPLIT|CLOCK_MARKS|L2_ATTN_MB|L2_MOE_MB|L2_NEXT_MB|L2_STRETCH_END_MB|L2_SKIP)=|^TF_EXL3_L2_' <<<"$envdump")
  [[ "$stray" == 0 ]] && pass "no rejected/experiment switches in container" \
    || bad "$stray rejected/experiment switch(es) present in container"
fi

avail=$(free -g | awk 'NR==2{print $7}')
if [[ -n "$avail" ]] && (( avail >= 6 )); then
  pass "avail ${avail} GiB (>= 6 GiB headroom)"
else
  bad "avail ${avail:-?} GiB is below the 6 GiB floor (production multi-stream min is 4.8 GiB)"
fi

echo
if (( fails == 0 )); then
  echo "RESULT: OK -- serving the baseline described in docs/CURRENT-BASELINE.md"
else
  echo "RESULT: $fails check(s) FAILED -- do not treat this machine as on-baseline"
fi
exit $(( fails > 0 ))