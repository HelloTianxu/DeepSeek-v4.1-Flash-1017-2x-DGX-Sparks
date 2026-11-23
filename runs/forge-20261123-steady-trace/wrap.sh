#!/usr/bin/env bash
set -u
nsys profile --trace=cuda-sw,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=graph --output=/forge-trace/trace "$@" &
child=$!
trap 'kill -TERM "$child" 2>/dev/null || true; wait "$child"; exit 0' TERM INT
wait "$child"
