#!/usr/bin/env bash
# Deploy the git baseline to the two Sparks and make it come up on every boot.
# RUN THIS FROM THE CHECKOUT (e.g. the Mac). It talks to the head over ssh; the head
# reaches the worker itself (the Mac cannot reach the worker directly).
#
#   tools/deploy-baseline.sh                 # sync code, materialize config, install unit, restart, self-check
#   RESTART=0 tools/deploy-baseline.sh       # do everything except restarting the engine
#   FORCE=1 tools/deploy-baseline.sh         # overwrite a local.sh that differs from the authoritative copy
#
# What it guarantees afterwards (docs/CURRENT-BASELINE.md §6):
#   scripts/local.sh == scripts/local.sh.authoritative   (the git-tracked best baseline config)
#   image base-alias present on BOTH nodes
#   user unit installed with the right WorkingDirectory, enabled, Linger=yes
#   engine healthy and serving that config
#
# It never deletes anything on the nodes and never touches local.sh backups.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
HEAD_SSH=${HEAD_SSH:-ztxjzh@192.168.1.118}
REMOTE_DIR=${REMOTE_DIR:-'~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks'}
IMAGE=${IMAGE:-tensorfold-dsv41:base-alias}
UNIT=dsv41-tensorfold.service

say() { printf '\n== %s ==\n' "$1"; }

say "1/6 sync deployment code"
# local.sh is machine-specific and gitignored: never sync it, it is materialized in step 2.
rsync -a --exclude 'local.sh' --exclude '__pycache__' --exclude '*.pyc' \
  "$REPO/start.sh" "$REPO/stop.sh" "$REPO/scripts" "$REPO/systemd" "$REPO/tools" \
  "$HEAD_SSH:$REMOTE_DIR/"
echo "  synced start.sh stop.sh scripts/ systemd/ tools/"

say "2/6 materialize the authoritative config"
ssh "$HEAD_SSH" "cd $REMOTE_DIR && \
  if [[ ! -f scripts/local.sh ]]; then \
    cp scripts/local.sh.authoritative scripts/local.sh && echo '  local.sh created from authoritative'; \
  elif diff -q scripts/local.sh scripts/local.sh.authoritative >/dev/null; then \
    echo '  local.sh already matches authoritative'; \
  elif [[ '${FORCE:-0}' == 1 ]]; then \
    cp scripts/local.sh scripts/local.sh.pre-deploy-bak && \
    cp scripts/local.sh.authoritative scripts/local.sh && \
    echo '  local.sh replaced (previous kept as local.sh.pre-deploy-bak)'; \
  else \
    echo '  REFUSING: local.sh differs from scripts/local.sh.authoritative:' >&2; \
    diff scripts/local.sh.authoritative scripts/local.sh | head -20 >&2; \
    echo '  re-run with FORCE=1 to adopt the git baseline' >&2; exit 1; \
  fi"

say "3/6 verify the image on both nodes"
ssh "$HEAD_SSH" "docker image inspect $IMAGE >/dev/null 2>&1 || { echo '  head image missing: $IMAGE' >&2; exit 1; }; \
  echo '  head ok'; \
  W=\$(sed -n 's/^WORKER=//p' $REMOTE_DIR/scripts/local.sh | head -1); \
  ssh -o BatchMode=yes \"\$W\" 'docker image inspect $IMAGE >/dev/null 2>&1' \
    || { echo \"  worker image missing: $IMAGE\" >&2; exit 1; }; \
  echo '  worker ok'"

say "4/6 install the user unit"
ssh "$HEAD_SSH" "cd $REMOTE_DIR && \
  mkdir -p ~/.config/systemd/user && \
  abs=\$(pwd) && \
  sed \"s|^WorkingDirectory=.*|WorkingDirectory=\$abs|\" systemd/$UNIT > ~/.config/systemd/user/$UNIT && \
  chmod +x scripts/wait-fabric.sh tools/*.sh 2>/dev/null; \
  systemctl --user daemon-reload && systemctl --user enable $UNIT >/dev/null 2>&1; \
  echo \"  WorkingDirectory=\$abs\"; \
  echo \"  enabled=\$(systemctl --user is-enabled $UNIT)\"; \
  L=\$(loginctl show-user \$USER --property=Linger --value); \
  echo \"  Linger=\$L\"; \
  [[ \"\$L\" == yes ]] || echo '  WARNING: Linger is not yes; run: loginctl enable-linger '\$USER >&2"

if [[ "${RESTART:-1}" == 1 ]]; then
  say "5/6 restart and wait for health"
  ssh "$HEAD_SSH" "cd $REMOTE_DIR && \
    systemctl --user stop $UNIT; sleep 8; \
    systemctl --user reset-failed $UNIT; \
    systemctl --user start $UNIT; \
    K=\$(cat ~/.local/state/dsv41-tensorfold/api-key); \
    for i in \$(seq 1 60); do \
      c=\$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8888/health -H \"Authorization: Bearer \$K\"); \
      [[ \"\$c\" == 200 ]] && { echo '  READY'; break; }; sleep 5; \
    done; \
    [[ \"\$c\" == 200 ]] || { echo '  engine did not become healthy' >&2; exit 1; }"
else
  say "5/6 restart skipped (RESTART=0)"
fi

say "6/6 self-check"
ssh "$HEAD_SSH" "cd $REMOTE_DIR && bash tools/check-boot.sh"

echo
echo "Deploy done. After any reboot, verify with: ssh $HEAD_SSH 'cd $REMOTE_DIR && bash tools/check-boot.sh'"