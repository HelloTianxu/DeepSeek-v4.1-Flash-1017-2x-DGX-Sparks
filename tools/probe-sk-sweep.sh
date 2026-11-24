#!/usr/bin/env bash
# E-SELF-07c：gate/up 的 K 分裂数 SK 单因子扫描（WK 固定 4，与生产一致）。
# SK*WK 必须整除 KT=320，且 chain=320/(SK*WK) 必须是 4 的倍数，才走同一个 KB=4 内核变体。
# 目的：让每个程序折叠更多 K 分裂（每 warp 更多独立 load 在飞），看达成 DRAM 带宽能否突破 ~154 GB/s。
set -uo pipefail
cd "$(dirname "$0")"
for SK in 4 1 2 5 8 20 4; do
  CH=$((320 / (SK * 4)))
  echo "--- SK=${SK} WK=4 chain=${CH} ---"
  python3 probe-layer.py --routing shared6 --which mma --sk "$SK" --wk 4 2>&1 \
    | grep -E '^mma|PROBE_OK|Error|error|Traceback|Check' | head -4
done
echo SWEEP_OK