#!/usr/bin/env bash
# E-SELF-04 source-side bit-identity gate.
# Rebuild ONLY experts_cb2.cu (our CB=2 served kernels) from the public upstream source with
# the recipe found in the tree, then compare its kernels' SASS against the kit cubin.
#
# Recipe provenance (all read, not guessed):
#   src/tensorfold/cuda/exl3/experts.py:129  extra_cuda_cflags=["-O3","-lineinfo"]
#   src/tensorfold/cuda/build.py:33          -gencode=arch=compute_121,code=sm_121
#   torch.utils.cpp_extension defaults       printed below and passed explicitly
#
# -cubin on a single .cu: no pybind module needed, so experts.cpp is not required.
set -eu
W=/work
S=$W/exl3src
test -s "$S/experts_cb2.cu" || { echo "FAIL: sources missing"; exit 1; }
test -s "$W/res-kit.txt" || { echo "FAIL: run cubin-roundtrip-gate.sh first"; exit 1; }

echo "== torch 默认 nvcc flags 与 include =="
TORCH=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
python3 -c "
from torch.utils import cpp_extension as C
print('  COMMON_NVCC_FLAGS:', C.COMMON_NVCC_FLAGS)
print('  torch:', '$TORCH')
"
INC="-I$TORCH/include -I$TORCH/include/torch/csrc/api/include -I/usr/include/python3.12"

# torch's cpp_extension passes these for every CUDA TU; exl3 adds -O3 -lineinfo on top.
FLAGS="-O3 -lineinfo -gencode=arch=compute_121,code=sm_121 -std=c++17 --expt-relaxed-constexpr
-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
-D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
-DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=probe"

echo "== 编译 experts_cb2.cu -> cubin =="
cd "$S"
T0=$SECONDS
set +e
nvcc -cubin $FLAGS $INC experts_cb2.cu -o "$W/cb2.cubin" > "$W/cb2-build.log" 2>&1
rc=$?
set -e
echo "  nvcc rc=$rc  用时 $((SECONDS-T0))s"
if [ $rc -ne 0 ]; then
  echo "  BUILD_FAILED — 前 30 行:"
  head -30 "$W/cb2-build.log"
  exit 1
fi
test -s "$W/cb2.cubin" || { echo FAIL_EMPTY_CUBIN; exit 1; }
echo "  cb2.cubin $(stat -c%s "$W/cb2.cubin") B"

echo "== 资源账 + SASS 对照 =="
cuobjdump -res-usage "$W/cb2.cubin" 2>/dev/null | awk '
  $1=="Function" && $2 ~ /tf_exl3x/ {name=$2; sub(/:$/,"",name); next}
  name!="" && $1 ~ /^REG:/ {print name, $1, $2, $3; name=""}' | sort > "$W/res-cb2.txt"
n=$(wc -l < "$W/res-cb2.txt")
echo "  自建 cb2 内核数: $n"
test "$n" -gt 0 || { echo FAIL_EMPTY_RESUSAGE; exit 1; }

# compare only the kernels this TU actually produced
python3 - "$W/res-kit.txt" "$W/res-cb2.txt" <<'PY'
import sys
kit = dict(l.split(' ', 1) for l in open(sys.argv[1]).read().strip().split('\n'))
new = dict(l.split(' ', 1) for l in open(sys.argv[2]).read().strip().split('\n'))
same = [k for k in new if k in kit and kit[k] == new[k]]
diff = [k for k in new if k in kit and kit[k] != new[k]]
miss = [k for k in new if k not in kit]
print('  RESUSAGE same=%d diff=%d not-in-kit=%d' % (len(same), len(diff), len(miss)))
for k in diff[:10]:
    print('    DIFF', k.split('kernel')[-1][:40], 'kit:', kit[k], '| new:', new[k])
for k in miss[:5]:
    print('    ONLY-NEW', k[:80])
PY

for V in 16 10; do
  NAME=$(awk -v v="ILi2ELi8ELi4ELi1ELi2ELi${V}ELi2ELb0" \
    'index($1,"grouped_rows_kernel") && index($1,v){print $1}' "$W/res-cb2.txt" | head -1)
  test -n "$NAME" || { echo "  HI=$V NOT_IN_OUR_BUILD"; continue; }
  for tag in kit cb2; do
    f=/hf/tensorfold-dsv41/kit-fdde3285/kit/cubins/experts_cb.cubin
    [ "$tag" = cb2 ] && f=$W/cb2.cubin
    cuobjdump -sass -fun "$NAME" "$f" 2>/dev/null \
      | sed -n 's@^[[:space:]]*/\*[0-9a-f]\{4\}\*/[[:space:]]\+\(.*\);.*$@\1@p' \
      | sed 's/[[:space:]]\+/ /g; s/[[:space:]]*$//' > "$W/g-$V-$tag.txt"
    test -s "$W/g-$V-$tag.txt" || { echo "  HI=$V FAIL_SASS_EMPTY($tag)"; exit 1; }
  done
  if cmp -s "$W/g-$V-kit.txt" "$W/g-$V-cb2.txt"; then
    echo "  HI=$V VERDICT IDENTICAL ($(wc -l < "$W/g-$V-kit.txt") instrs)"
  else
    echo "  HI=$V VERDICT DIFFERS ($(diff "$W/g-$V-kit.txt" "$W/g-$V-cb2.txt" | grep -c '^[<>]') lines)"
    diff "$W/g-$V-kit.txt" "$W/g-$V-cb2.txt" | head -8
  fi
done
echo SRC_GATE_DONE