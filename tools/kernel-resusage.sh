#!/usr/bin/env bash
# Register / shared-memory / spill report for the kernels that actually matter. RUN ON THE HEAD NODE.
#
# Why this exists: E-SELF-02 was proposed on a documented static account ("down 255 regs/thread
# => 16.7% occupancy") that turned out to be wrong AND to describe another family's kernels.
# Never reason about occupancy from a number in a doc -- read it out of the binary.
# See runs/forge-20261124-self-occupancy/REPORT.md.
#
#   tools/kernel-resusage.sh            # kit cubins (our served exl3 expert path) + engine kernels
#   KIT_ONLY=1 tools/kernel-resusage.sh # just the cubins (fast, no compile)
#
# Read-only: -Xptxas -v and cuobjdump only print; nothing is installed, launched or rebuilt.
set -uo pipefail

BASE_IMAGE=${BASE_IMAGE:-tensorfold-dsv41:exl3-tp2-8348338a8d90}
KIT_REF=${KIT_REF:-fdde3285}
CUBINS=${CUBINS:-$HOME/.cache/huggingface/tensorfold-dsv41/kit-$KIT_REF/kit/cubins}
SRC=${SRC:-$HOME/build-alias}
SM=${SM:-121}

if [[ ! -d $CUBINS ]]; then
  echo "FAIL: cubin dir not found: $CUBINS" >&2
  exit 1
fi

# Our served expert kernels live in the kit as prebuilt cubins -- there is no .cu source anywhere,
# so cuobjdump is the only way to see their resource usage.
echo "== kit cubins (tf_exl3x: the served dsv41 expert path) =="
docker run --rm --memory=8g --memory-swap=8g --entrypoint bash \
  -v "$CUBINS":/cubins:ro "$BASE_IMAGE" -c '
    for c in experts_cb experts linear_grouped linear; do
      [ -f /cubins/$c.cubin ] || continue
      echo "##### $c.cubin"
      cuobjdump -res-usage /cubins/$c.cubin 2>/dev/null | awk "
        /Function _ZN8tf_exl3x/ { name=\$2; sub(/^_ZN8tf_exl3x/,\"\",name); next }
        /REG:/ && name != \"\" {
          reg=\"\"; stk=\"\"; shm=\"\";
          for(i=1;i<=NF;i++){
            if(\$i ~ /^REG:/){reg=\$i}
            if(\$i ~ /^STACK:/){stk=\$i}
            if(\$i ~ /^SHARED:/){shm=\$i}
          }
          printf \"  %-58s %s %s %s\n\", substr(name,1,58), reg, stk, shm; name=\"\"
        }"
    done
  ' 2>/dev/null | grep -E '#####|REG:'

if [[ ${KIT_ONLY:-0} == 1 ]]; then
  echo "RESULT: cubin report only (KIT_ONLY=1)"
  exit 0
fi

if [[ ! -d $SRC/zig/kernels/cuda ]]; then
  echo "SKIP engine kernels: source tree not found at $SRC (set SRC=...)" >&2
  exit 0
fi

# Engine-side kernels: compile each alone with -Xptxas -v. NOTE these are the Nemotron family's
# experts/qmm/lane_gemv -- dsv41 does not call them. Kept so the family distinction stays visible.
echo
echo "== engine kernels (nvcc -Xptxas -v; Nemotron family, NOT the dsv41 path) =="
docker run --rm --memory=8g --memory-swap=8g --entrypoint bash \
  -v "$SRC":/tensorfold:ro "$BASE_IMAGE" -c "
    cd /tensorfold
    for k in experts experts_prefill qmm_group qmm_prefill lane_gemv experts_pack; do
      echo \"##### \$k\"
      nvcc -fatbin \
        -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ \
        -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
        --expt-relaxed-constexpr -std=c++20 -O3 -Xptxas -v \
        -gencode=arch=compute_$SM,code=sm_$SM \
        -o /tmp/\$k.fatbin zig/kernels/cuda/\$k.cu 2>&1 \
        | grep -E 'Compiling entry|Used [0-9]+ registers|spill' \
        | sed 's/ptxas info    : //' | cut -c1-170
    done
  " 2>/dev/null | grep -E '#####|Compiling entry|Used|spill'

echo
echo "Reminder: static occupancy = floor(65536 / (regs * threads)) blocks * (threads/32) warps / 48."
echo "If achieved occupancy is BELOW that ceiling, registers are not the limiter -- see REPORT §3."