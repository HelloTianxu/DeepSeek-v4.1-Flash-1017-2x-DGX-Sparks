#include <cstdio>
#include <cuda_runtime.h>
int main() {
    int optin = 0, smem_sm = 0, regs_sm = 0, warps_sm = 0, sm = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
    cudaDeviceGetAttribute(&regs_sm, cudaDevAttrMaxRegistersPerMultiprocessor, 0);
    cudaDeviceGetAttribute(&warps_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    printf("MaxSharedMemoryPerBlockOptin = %d B\n", optin);
    printf("SharedMemoryPerMultiprocessor = %d B\n", smem_sm);
    printf("RegistersPerMultiprocessor = %d\n", regs_sm);
    printf("MaxThreadsPerMultiProcessor = %d\n", warps_sm);
    printf("SM = %d\n", sm);
    // mma2 R=64 KB=4 k2=6 时每级 15360 B
    for (int S = 3; S <= 8; ++S) {
        int bytes = S * 64 * (4 * 16 + 8) * 2 + S * 4 * 16 * 4 * 6 * 4;
        printf("  S=%d  smem=%6d B  +512=%6d  %s\n", S, bytes, bytes + 512,
               (bytes + 512 <= optin) ? "FIT" : "REJECT");
    }
    return 0;
}