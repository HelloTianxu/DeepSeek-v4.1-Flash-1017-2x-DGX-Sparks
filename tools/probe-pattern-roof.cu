// E-SELF-09c：访存形状的屋顶标定 —— 把「卡车路本身限速」与「内核递货低效」分开。
//
// E-SELF-09 已证：S∈{3,4,5,6}、MINB∈{1,2} 全部落在 160–165 GB/s，杠杆榨干。
// 本文件三个内核读**完全相同的总字节**，只差访存形状：
//   stream  : 全缓冲顺序流式读 ⇒ 本进程本时刻的 streaming 屋顶（E-DRAM-01 口径复标）
//   pattern : mma2 的真实形状 —— 每程序 KT 个 1536 B 块，块间跨距 NTILES*96 B，无解码无 MMA
//   slab    : 同字节同程序数同早退，但每程序读**连续** slab ⇒ 若权重布局改成程序私有连续能到多少
// pattern ≈ mma2 实测 ⇒ 已到该形状的硬件上限，杠杆在布局不在流水；
// slab ≫ pattern ⇒ 布局（跨距）就是当下的单点卡口。
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void stream_read(const uint4* __restrict__ p, long n, uint32_t* __restrict__ out) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    uint32_t acc = 0;
    for (; i < n; i += stride) {
        uint4 v = __ldg(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    out[blockIdx.x] = acc;
}

__device__ __forceinline__ int prog(const int* ucount, const int* members, const int* work,
                                    int maxm, int nexp, int R, int* u_out) {
    int zz = blockIdx.z % nexp;
    int u = work[2 * zz];
    if (u >= ucount[0]) return 0;
    int mb = work[2 * zz + 1];
    if (members[(long)u * maxm + (long)mb * R] < 0) return 0;
    *u_out = u;
    return 1;
}

__global__ void pattern_read(const int64_t* __restrict__ TP, const int* __restrict__ uids,
                             const int* __restrict__ ucount, const int* __restrict__ members,
                             const int* __restrict__ work, int maxm, int nexp,
                             int KT, int NTILES, int TW, int M2T, int R,
                             uint32_t* __restrict__ out) {
    int u;
    if (!prog(ucount, members, work, maxm, nexp, R, &u)) return;
    const uint4* T = reinterpret_cast<const uint4*>(TP[uids[u]]);
    const int CH = M2T * TW / 4;              // 96 uint4 = 1536 B 一个 k tile
    const int STR = NTILES * TW / 4;          // k tile 间跨距（uint4）
    const int off = blockIdx.y * M2T * (TW / 4);
    const int total = KT * CH;
    uint32_t acc = 0;
    for (int c = threadIdx.x; c < total; c += blockDim.x) {
        const int kt = c / CH, q = c % CH;
        uint4 v = __ldg(T + (long)kt * STR + off + q);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    out[blockIdx.z * gridDim.y + blockIdx.y] = acc;
}

// mma2 的**完整** staged 字节集：权重 + 激活，按 stage 顺序，无 shared/栅栏/ldmatrix。
// 前一版只搬权重（3.43 GiB），漏掉激活被 9 个列块重复 staging 的部分 ⇒ 对比不成立，此处补齐。
__global__ void pattern_full(const int64_t* __restrict__ TP, const int* __restrict__ uids,
                             const int* __restrict__ ucount, const int* __restrict__ members,
                             const int* __restrict__ work, const half* __restrict__ X,
                             int maxm, int nexp, int KT, int NTILES, int TW, int M2T, int R,
                             int K, int slots, int KB, uint32_t* __restrict__ out) {
    int u;
    if (!prog(ucount, members, work, maxm, nexp, R, &u)) return;
    const uint4* T = reinterpret_cast<const uint4*>(TP[uids[u]]);
    const int CH = M2T * TW / 4;
    const int STR = NTILES * TW / 4;
    const int off = blockIdx.y * M2T * (TW / 4);
    const int zz = blockIdx.z % nexp;
    const int mb = work[2 * zz + 1];
    const int NS = KT / KB;
    const int CPR = KB * 2;                       // 16B chunks a staged row
    // 与内核一致：只搬 live 行（16 行一组，整组空则跳过）
    int live = 0;
    for (int g = 0; g < R / 16; ++g) {
        int any = 0;
        for (int r = 0; r < 16; ++r)
            if (members[(long)u * maxm + (long)mb * R + g * 16 + r] >= 0) any = 1;
        if (any) live = g + 1;
    }
    uint32_t acc = 0;
    for (int st = 0; st < NS; ++st) {
        for (int c = threadIdx.x; c < KB * CH; c += blockDim.x) {
            const int kk = c / CH, q = c % CH;
            uint4 v = __ldg(T + (long)(st * KB + kk) * STR + off + q);
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
        const int kb = st * KB * 16;
        for (int c = threadIdx.x; c < live * 16 * CPR; c += blockDim.x) {
            const int r = c / CPR, q = c % CPR;
            const int code = members[(long)u * maxm + (long)mb * R + r];
            const int p = code >= 0 ? (code >> 5) * slots + (code & 31) : 0;
            const uint4* src = reinterpret_cast<const uint4*>(X + (long)p * K + kb + q * 8);
            uint4 v = __ldg(src);
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
    }
    out[blockIdx.z * gridDim.y + blockIdx.y] = acc;
}

__global__ void slab_read(const int64_t* __restrict__ TP, const int* __restrict__ uids,
                          const int* __restrict__ ucount, const int* __restrict__ members,
                          const int* __restrict__ work, int maxm, int nexp,
                          int KT, int NTILES, int TW, int M2T, int R,
                          uint32_t* __restrict__ out) {
    int u;
    if (!prog(ucount, members, work, maxm, nexp, R, &u)) return;
    const uint4* T = reinterpret_cast<const uint4*>(TP[uids[u]]);
    const int CH = M2T * TW / 4;
    const long total = (long)KT * CH;
    const long base = (long)blockIdx.y * total;   // 每列块一段连续，9 段刚好铺满该专家
    uint32_t acc = 0;
    for (long c = threadIdx.x; c < total; c += blockDim.x) {
        uint4 v = __ldg(T + base + c);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    out[blockIdx.z * gridDim.y + blockIdx.y] = acc;
}

void launch_stream(const at::Tensor& buf, at::Tensor out, int64_t blocks) {
    long n = buf.numel() / 4;                   // int32 → uint4
    auto s = at::cuda::getCurrentCUDAStream();
    stream_read<<<(int)blocks, 256, 0, s>>>(
        reinterpret_cast<const uint4*>(buf.data_ptr()), n, (uint32_t*)out.data_ptr());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename F>
void launch_grid(F fn, const at::Tensor& TP, const at::Tensor& uids, const at::Tensor& ucount,
                 const at::Tensor& members, const at::Tensor& work, at::Tensor out,
                 int64_t KT, int64_t NTILES, int64_t TW, int64_t M2T, int64_t R,
                 int64_t ny, int64_t nz) {
    dim3 grid(1, (unsigned)ny, (unsigned)nz);
    auto s = at::cuda::getCurrentCUDAStream();
    fn<<<grid, 256, 0, s>>>(TP.data_ptr<int64_t>(), uids.data_ptr<int>(), ucount.data_ptr<int>(),
                            members.data_ptr<int>(), work.data_ptr<int>(), (int)members.size(1),
                            (int)(nz / 2), (int)KT, (int)NTILES, (int)TW, (int)M2T, (int)R,
                            (uint32_t*)out.data_ptr());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_pattern(const at::Tensor& TP, const at::Tensor& uids, const at::Tensor& ucount,
                    const at::Tensor& members, const at::Tensor& work, at::Tensor out,
                    int64_t KT, int64_t NTILES, int64_t TW, int64_t M2T, int64_t R,
                    int64_t ny, int64_t nz) {
    launch_grid(pattern_read, TP, uids, ucount, members, work, out, KT, NTILES, TW, M2T, R, ny, nz);
}
void launch_slab(const at::Tensor& TP, const at::Tensor& uids, const at::Tensor& ucount,
                 const at::Tensor& members, const at::Tensor& work, at::Tensor out,
                 int64_t KT, int64_t NTILES, int64_t TW, int64_t M2T, int64_t R,
                 int64_t ny, int64_t nz) {
    launch_grid(slab_read, TP, uids, ucount, members, work, out, KT, NTILES, TW, M2T, R, ny, nz);
}

void launch_full(const at::Tensor& TP, const at::Tensor& uids, const at::Tensor& ucount,
                 const at::Tensor& members, const at::Tensor& work, const at::Tensor& X,
                 at::Tensor out, int64_t KT, int64_t NTILES, int64_t TW, int64_t M2T, int64_t R,
                 int64_t ny, int64_t nz, int64_t K, int64_t slots, int64_t KB) {
    dim3 grid(1, (unsigned)ny, (unsigned)nz);
    auto s = at::cuda::getCurrentCUDAStream();
    pattern_full<<<grid, 256, 0, s>>>(TP.data_ptr<int64_t>(), uids.data_ptr<int>(), ucount.data_ptr<int>(),
                                      members.data_ptr<int>(), work.data_ptr<int>(),
                                      reinterpret_cast<const half*>(X.data_ptr()),
                                      (int)members.size(1), (int)(nz / 2), (int)KT, (int)NTILES,
                                      (int)TW, (int)M2T, (int)R, (int)K, (int)slots, (int)KB,
                                      (uint32_t*)out.data_ptr());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stream", &launch_stream);
    m.def("pattern", &launch_pattern);
    m.def("slab", &launch_slab);
    m.def("full", &launch_full);
}