// E-FEED-02 机制判决：24 MiB HtoD 在 GPU 有负载时，pageable 与 pinned 的 host 侧 API 时长。
//
// 为什么必须带负载、必须用真实尺寸：
//   历史 pageable-copy-gate 用 4096 B 空载测得 pageable 2.284 µs vs pinned 1.831 µs，判「可忽略」。
//   但生产是 24 MiB 且 stream 上排着几百 ms 的 kernel：pageable 源要 driver 分块 staging，
//   host 被 GPU 消费速度反向限速（实测 375-642 ms，同期 GPU 忙 100%）。小包空载暴露不出这个效应。
//
// 判据：
//   pageable 的 API 时长 ≈ spin 时长，而 pinned 的 ≈ 纯传输时间  ⇒ pinned 确实解除节流；
//   两者都 ≈ spin 时长                                        ⇒ GB10 统一内存下 pinned 不解节流。
// 前者成立且端到端无收益 ⇒ 「upload 节流导致 GPU 空隙」假说被否证（而非实现无效）。
//
// 编译: nvcc -O3 -arch=sm_121 -o /tmp/pin-gate pin-gate.cu
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ printf("ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 2; } }while(0)

__global__ void spin(unsigned long long n, float* out) {
    float a = 1.0f;
    for (unsigned long long i = 0; i < n; ++i) a = a * 1.0000001f + 1e-9f;
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = a;
}

static double med(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    const size_t B = argc > 1 ? (size_t)atoll(argv[1]) : (size_t)2048 * 6144 * 2; // 24 MiB
    const unsigned long long N = argc > 2 ? strtoull(argv[2], 0, 10) : 40000000ULL;
    const int REPS = 5;

    void* dst = nullptr;
    float* sink = nullptr;
    CK(cudaMalloc(&dst, B));
    CK(cudaMalloc(&sink, sizeof(float)));
    void* pag = malloc(B);
    void* pin = nullptr;
    CK(cudaHostAlloc(&pin, B, cudaHostAllocPortable));
    if (!pag) { printf("ERR malloc\n"); return 2; }
    memset(pag, 1, B);
    memset(pin, 1, B);

    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    // 标定 spin 时长（device event，不用 host 时钟）
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    spin<<<48, 256, 0, s>>>(N, sink);
    CK(cudaEventRecord(e0, s));
    spin<<<48, 256, 0, s>>>(N, sink);
    CK(cudaEventRecord(e1, s));
    CK(cudaStreamSynchronize(s));
    float spin_ms = 0;
    CK(cudaEventElapsedTime(&spin_ms, e0, e1));

    printf("{\"bytes\": %zu, \"MiB\": %.2f, \"spin_ms\": %.3f, \"reps\": %d,\n", B, B / 1048576.0, spin_ms, REPS);

    for (int kind = 0; kind < 2; ++kind) {
        void* src = kind ? pin : pag;
        std::vector<double> api, tot;
        for (int r = 0; r < REPS; ++r) {
            // 排一个长 kernel，让 stream 非空
            spin<<<48, 256, 0, s>>>(N, sink);
            auto a = std::chrono::steady_clock::now();
            CK(cudaMemcpyAsync(dst, src, B, cudaMemcpyHostToDevice, s));
            auto b = std::chrono::steady_clock::now();
            CK(cudaStreamSynchronize(s));
            auto c = std::chrono::steady_clock::now();
            api.push_back(std::chrono::duration<double, std::milli>(b - a).count());
            tot.push_back(std::chrono::duration<double, std::milli>(c - a).count());
        }
        printf("  \"%s\": {\"api_ms_median\": %.3f, \"api_plus_sync_ms_median\": %.3f}%s\n",
               kind ? "pinned" : "pageable", med(api), med(tot), kind ? "" : ",");
    }
    printf("}\n");

    CK(cudaFreeHost(pin));
    free(pag);
    CK(cudaFree(dst));
    CK(cudaFree(sink));
    CK(cudaStreamDestroy(s));
    return 0;
}