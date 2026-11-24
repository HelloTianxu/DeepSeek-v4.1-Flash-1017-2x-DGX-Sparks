#!/usr/bin/env bash
# Bounded wait for the worker node before the engine starts.
#
# Why: the user service is pulled up by linger at boot, which can happen before the fabric
# interface and the worker's docker daemon are ready. start.sh would then fail to launch rank 1.
# This runs as ExecStartPre so the engine waits instead of burning a StartLimit attempt.
#
# FABRIC_WAIT_S bounds the wait (default 300s). Exits non-zero on timeout so systemd retries.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

if [[ ! -f scripts/local.sh ]]; then
  echo "wait-fabric: scripts/local.sh is missing. Materialize it from scripts/local.sh.authoritative" >&2
  echo "             (tools/deploy-baseline.sh does this). Refusing to start." >&2
  exit 1
fi

WORKER=$(sed -n 's/^WORKER=//p' scripts/local.sh | head -1 | tr -d '"'"'"'')
if [[ -z "$WORKER" ]]; then
  echo "wait-fabric: no WORKER= line in scripts/local.sh" >&2
  exit 1
fi

limit=${FABRIC_WAIT_S:-300}
deadline=$((SECONDS + limit))
while (( SECONDS < deadline )); do
  if ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
       "$WORKER" 'docker info >/dev/null 2>&1' 2>/dev/null; then
    echo "wait-fabric: worker $WORKER reachable and docker responsive"
    exit 0
  fi
  sleep 5
done

echo "wait-fabric: worker $WORKER not ready after ${limit}s" >&2
exit 1