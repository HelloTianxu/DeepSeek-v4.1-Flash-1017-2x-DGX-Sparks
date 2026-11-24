#!/usr/bin/env bash
# E-SELF-04 route B, PTX level: keep the served kernel's SHAPE untouched (NT=8, G=2, PF=1,
# FOLD=0) and only raise the compiler's minimum-blocks requirement, then read what ptxas
# does to REG / STACK. This isolates the register ceiling from route A's shape change
# (which doubled decode work by raising MG 64->128).
#
# Occupancy arithmetic on GB10 (65536 regs/SM, 48 warps/SM, 128 threads/block = 4 warps):
#   2 blocks -> 65536/256 = 256 regs/thread  (current: REG 255, 8 warps = 16.7%)
#   3 blocks -> 65536/384 = 170 -> REG <= 168 (12 warps = 25.0%)
#   4 blocks -> 65536/512 = 128 regs/thread  (16 warps = 33.3%)
#
# Compile-only. No GPU, no engine rebuild, ~20 s per variant.
set -eu
W=/work
K=/hf/tensorfold-dsv41/kit-fdde3285/kit/cubins
test -s "$W/kit.ptx" || { echo "FAIL: run cubin-roundtrip-gate.sh first"; exit 1; }

echo "== 基线（未改动）served 两件 =="
grep -E 'grouped_rows_kernelILi2ELi8ELi4ELi1ELi2ELi(16|10)ELi2ELb0' "$W/res-kit.txt" \
  | sed 's/_ZN8tf_exl3x19grouped_rows_kernel//; s/EEEv.*//' | sed 's/^/  /'

for MB in 3 4; do
  echo
  echo "== .minnctapersm $MB （目标 REG <= $([ $MB = 3 ] && echo 168 || echo 128)）=="
  python3 - "$W/kit.ptx" "$W/mb$MB.ptx" "$MB" <<'PY'
import sys
src, dst, mb = sys.argv[1], sys.argv[2], sys.argv[3]
L = open(src).read().split('\n')
targets = ('grouped_rows_kernelILi2ELi8ELi4ELi1ELi2ELi16ELi2ELb0',
           'grouped_rows_kernelILi2ELi8ELi4ELi1ELi2ELi10ELi2ELb0')
out, i, hits = [], 0, 0
while i < len(L):
    out.append(L[i])
    if '.entry' in L[i] and any(t in L[i] for t in targets):
        # copy params up to the .maxntid line, then inject .minnctapersm right after it
        i += 1
        while i < len(L):
            out.append(L[i])
            if L[i].strip().startswith('.maxntid'):
                out.append('.minnctapersm ' + mb)
                hits += 1
                break
            if L[i].strip() == '{':
                sys.exit('FAIL: hit body before .maxntid')
            i += 1
    i += 1
assert hits == 2, 'FAIL: expected 2 served entries, patched %d' % hits
open(dst, 'w').write('\n'.join(out))
print('  patched entries:', hits)
PY
  ptxas -arch=sm_121 -o "$W/mb$MB.cubin" "$W/mb$MB.ptx" 2> "$W/mb$MB.log" || {
    echo "  PTXAS_FAILED"; tail -8 "$W/mb$MB.log"; continue; }
  cuobjdump -res-usage "$W/mb$MB.cubin" 2>/dev/null | awk '
    $1=="Function" && $2 ~ /tf_exl3x/ {name=$2; sub(/:$/,"",name); next}
    name!="" && $1 ~ /^REG:/ {print name, $1, $2, $3; name=""}' | sort > "$W/res-mb$MB.txt"
  test -s "$W/res-mb$MB.txt" || { echo "  FAIL_EMPTY_RESUSAGE"; continue; }
  echo "  served 两件结果:"
  grep -E 'grouped_rows_kernelILi2ELi8ELi4ELi1ELi2ELi(16|10)ELi2ELb0' "$W/res-mb$MB.txt" \
    | sed 's/_ZN8tf_exl3x19grouped_rows_kernel//; s/EEEv.*//' | sed 's/^/    /'
  echo "  其余 84 个内核是否被波及:"
  if diff <(grep -vE 'grouped_rows_kernelILi2ELi8ELi4ELi1ELi2ELi(16|10)ELi2ELb0' "$W/res-kit.txt") \
          <(grep -vE 'grouped_rows_kernelILi2ELi8ELi4ELi1ELi2ELi(16|10)ELi2ELb0' "$W/res-mb$MB.txt") >/dev/null; then
    echo "    OTHERS_UNCHANGED"
  else
    echo "    OTHERS_CHANGED (需查)"
  fi
  echo "  全内核 SASS 差异行数（应只集中在 served 两件）:"
  cuobjdump -sass "$W/mb$MB.cubin" 2>/dev/null \
    | sed -n 's@^[[:space:]]*/\*[0-9a-f]\{4\}\*/[[:space:]]\+\(.*\);.*$@\1@p' \
    | sed 's/[[:space:]]\+/ /g; s/[[:space:]]*$//' > "$W/sass-mb$MB.txt"
  echo "    $(diff "$W/sass-all-kit.txt" "$W/sass-mb$MB.txt" | grep -c '^[<>]') lines"
done
echo PROBE_DONE