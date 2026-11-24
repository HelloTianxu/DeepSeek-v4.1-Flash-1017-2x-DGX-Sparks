// E-SELF-04b compile-time probe: hold G*NT = 16 so the accumulator budget stays at
// G*NT*8 = 128 fp32 registers and the program count stays constant, while the weight-decode
// redundancy factor MG = ceil(maxm / (16*G)) falls as G rises.
//
// Why this point is untested: route A measured (G=1,NT=8) and (G=2,NT=4), both of which
// RAISE the program count (MG up, or more column blocks at unchanged MG). Along G*NT=16
// the column-block count rises exactly as MG falls, so programs = (N/(NT*16)) * MG is
// invariant and only the decode work changes. Total tile-decodes = (N/16)*KT*MG.
//
// Known cost that rises with G: the warp-sum reduction runs 2 __syncthreads() per g,
// so barriers per program = 2G. The probe reads REG/STACK/SHARED; the A/B reads the trade.
#include "experts_grouped.cuh"

#define PROBE(NT, G, PF, HI)                                                                  \
    template __global__ void tf_exl3x::grouped_rows_kernel<2, NT, 4, PF, 2, HI, G, false>(    \
        const half*, const half*, const int64_t*, const int64_t*, const int*, const int*,     \
        const int*, const int*, const int*, float*, int, int, int, int, int, int, int,        \
        const int*);

PROBE(8, 2, 1, 16)   // baseline reference, must reproduce REG:255 STACK:0 SHARED:33920
PROBE(4, 4, 1, 16)   // decode /2, barriers x2
PROBE(2, 8, 1, 16)   // decode /4, barriers x4
PROBE(1, 16, 1, 16)  // decode /8, barriers x8
PROBE(4, 4, 2, 16)   // same shape, deeper prefetch (PF=2)
PROBE(4, 4, 1, 10)   // the other served k2 range