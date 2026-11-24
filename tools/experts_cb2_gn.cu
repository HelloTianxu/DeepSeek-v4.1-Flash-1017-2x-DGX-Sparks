// E-SELF-04b served cubin: the unmodified experts_cb2.cu superset, plus the G*NT=16 shape family
// the compile-time pre-screen cleared. One build serves every arm, so an A/B is an env flip and a
// restart, never a rebuild (E-SELF-02's pattern).
//
// The baseline instantiations are untouched; the added ones are new template arguments, so the
// served G=2 kernels must still come out bit-identical to the kit (verified by exl3-source-gate.sh).
#include "experts_cb2.cu"

// NT = 16/G keeps the accumulator at G*NT*8 = 128 fp32 registers and the program count invariant;
// MG = ceil(maxm/(16*G)) is the weight-decode redundancy that falls. Both served k2 ranges (HI 10
// and 16) are needed: range() buckets (2..10) and (2..16) separately, and the loader fills both.
#define GN(NT, G)                                                                              \
    template __global__ void tf_exl3x::grouped_rows_kernel<2, NT, 4, 1, 2, 16, G, false>(      \
        const half*, const half*, const int64_t*, const int64_t*, const int*, const int*,      \
        const int*, const int*, const int*, float*, int, int, int, int, int, int, int,         \
        const int*);                                                                           \
    template __global__ void tf_exl3x::grouped_rows_kernel<2, NT, 4, 1, 2, 10, G, false>(      \
        const half*, const half*, const int64_t*, const int64_t*, const int*, const int*,      \
        const int*, const int*, const int*, float*, int, int, int, int, int, int, int,         \
        const int*);

GN(4, 4)   // spill-free: REG:255 STACK:0 SHARED:17664 — decode work halved
GN(2, 8)   // second arm: STACK:72 — decode work quartered, small spill

// E-SELF-04b §5.3 factorial cell (NT=4, G=2): decode work unchanged, activation reads doubled.
// The kit only carries this shape at PF=2 (REG:168); PF=1 is needed to hold prefetch constant
// across the four cells, so it is instantiated here.
GN(4, 2)