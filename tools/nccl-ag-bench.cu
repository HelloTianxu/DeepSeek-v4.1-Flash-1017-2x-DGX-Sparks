// E-COMM-01 通信屋顶标定：直调 ncclAllGather，扫行数 x dtype。
// 用法: nccl-ag-bench RANK MASTER_IP PORT
// 与生产同路径（NCCL + IB + GDR），显存约 126 MB，可与生产并存。
// 计时：CUDA event 包住 reps 次调用；先 10 次预热。正确性：小整数图案，fp32/bf16 均精确可表示。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <nccl.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
#define NK(x) do{ ncclResult_t r=(x); if(r!=ncclSuccess){ \
  fprintf(stderr,"NCCL %s @%d: %s\n",#x,__LINE__,ncclGetErrorString(r)); exit(1);} }while(0)

static const int HIDDEN = 5120;
static const int MAXROWS = 2048;

// 图案：<=127 的整数，bf16(8 位尾数) 与 fp32 都精确
static inline float pat(int rank, int i) { return (float)((rank * 16 + (i % 16)) % 128); }

static void sendAll(int fd, const void* p, size_t n) {
  const char* b = (const char*)p; size_t off = 0;
  while (off < n) { ssize_t k = send(fd, b + off, n - off, 0); if (k <= 0) { perror("send"); exit(1); } off += (size_t)k; }
}
static void recvAll(int fd, void* p, size_t n) {
  char* b = (char*)p; size_t off = 0;
  while (off < n) { ssize_t k = recv(fd, b + off, n - off, 0); if (k <= 0) { perror("recv"); exit(1); } off += (size_t)k; }
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: nccl-ag-bench RANK MASTER_IP PORT\n"); return 2; }
  int rank = atoi(argv[1]);
  const char* ip = argv[2];
  int port = atoi(argv[3]);

  // --- bootstrap: rank0 分发 ncclUniqueId ---
  ncclUniqueId id;
  int fd = -1;
  if (rank == 0) {
    NK(ncclGetUniqueId(&id));
    int ls = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = INADDR_ANY;
    if (bind(ls, (sockaddr*)&a, sizeof(a)) < 0) { perror("bind"); return 1; }
    if (listen(ls, 4) < 0) { perror("listen"); return 1; }
    fd = accept(ls, nullptr, nullptr);
    if (fd < 0) { perror("accept"); return 1; }
    sendAll(fd, &id, sizeof(id));
    close(ls);
  } else {
    fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &a.sin_addr) != 1) { fprintf(stderr, "bad ip\n"); return 1; }
    for (int t = 0; t < 120; ++t) { if (connect(fd, (sockaddr*)&a, sizeof(a)) == 0) break; sleep(1); if (t == 119) { fprintf(stderr, "connect timeout\n"); return 1; } }
    recvAll(fd, &id, sizeof(id));
  }

  CK(cudaSetDevice(0));
  ncclComm_t comm;
  NK(ncclCommInitRank(&comm, 2, id, rank));
  cudaStream_t s; CK(cudaStreamCreate(&s));

  size_t maxElem = (size_t)MAXROWS * HIDDEN;
  void *dSend = nullptr, *dRecv = nullptr;
  CK(cudaMalloc(&dSend, maxElem * 4));
  CK(cudaMalloc(&dRecv, maxElem * 2 * 4));
  std::vector<float> h(maxElem * 2);
  std::vector<float> hd(maxElem * 2);  // 下载专用，保持 h 为本 rank 原始图案
  for (size_t i = 0; i < maxElem; ++i) h[i] = pat(rank, (int)i);
  CK(cudaMemcpy(dSend, h.data(), maxElem * 4, cudaMemcpyHostToDevice));

  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  printf("{\"rank\": %d", rank);

  const int rowsList[] = {24, 128, 512, 1024, 1280, 1536, 1792, 2048};
  for (int dt = 0; dt < 2; ++dt) {           // 0=fp32, 1=bf16
    ncclDataType_t ty = dt ? ncclBfloat16 : ncclFloat;
    int esz = dt ? 2 : 4;
    for (int rows : rowsList) {
      size_t cnt = (size_t)rows * HIDDEN;
      // 准备该 dtype 的发送图案
      if (dt) {
        std::vector<__nv_bfloat16> b(cnt);
        for (size_t i = 0; i < cnt; ++i) b[i] = __float2bfloat16(pat(rank, (int)i));
        CK(cudaMemcpy(dSend, b.data(), cnt * 2, cudaMemcpyHostToDevice));
      } else {
        CK(cudaMemcpy(dSend, h.data(), cnt * 4, cudaMemcpyHostToDevice));
      }
      CK(cudaMemsetAsync(dRecv, 0, cnt * 2 * 4, s));
      // 正确性
      NK(ncclAllGather(dSend, dRecv, cnt, ty, comm, s));
      CK(cudaStreamSynchronize(s));
      CK(cudaMemcpy(hd.data(), dRecv, cnt * 2 * esz, cudaMemcpyDeviceToHost));
      bool ok = true;
      for (int r = 0; r < 2 && ok; ++r) for (size_t i = 0; i < cnt; ++i) {
        float got = dt ? __bfloat162float(((const __nv_bfloat16*)hd.data())[r * cnt + i]) : hd[r * cnt + i];
        if (got != pat(r, (int)i)) { ok = false; break; }
      }
      const int reps = 50;
      for (int w = 0; w < 10; ++w) NK(ncclAllGather(dSend, dRecv, cnt, ty, comm, s));
      CK(cudaEventRecord(e0, s));
      for (int k = 0; k < reps; ++k) NK(ncclAllGather(dSend, dRecv, cnt, ty, comm, s));
      CK(cudaEventRecord(e1, s));
      CK(cudaEventSynchronize(e1));
      float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
      double us = ms * 1000.0 / reps;
      double bytes = (double)cnt * esz;
      printf(", \"%s_%d\": {\"us\": %.1f, \"bytes\": %.0f, \"GBs\": %.2f, \"ok\": %s}",
             dt ? "bf16" : "f32", rows, us, bytes, bytes / (us * 1e-6) / 1e9, ok ? "true" : "false");
      fflush(stdout);
    }
  }

  // ---- 分块实验：2048 行 fp32 拆成 S 个子 gather ----
  // pure = 只发子 gather（上界，忽略落地布局代价）；copy = 子 gather 落临时缓冲再拷进 [2,2048,D] 正确位置
  {
    void* dTmp = nullptr; CK(cudaMalloc(&dTmp, maxElem * 2 * 4));
    const int rows = 2048; const size_t cnt = (size_t)rows * HIDDEN;
    CK(cudaMemcpy(dSend, h.data(), cnt * 4, cudaMemcpyHostToDevice));
    for (int S : {2, 4, 8}) {
      const int sub = rows / S; const size_t scnt = (size_t)sub * HIDDEN;
      for (int mode = 0; mode < 2; ++mode) {
        auto issue = [&](void) {
          for (int k = 0; k < S; ++k) {
            NK(ncclAllGather((char*)dSend + (size_t)k * scnt * 4, dTmp, scnt, ncclFloat, comm, s));
            if (mode) for (int r = 0; r < 2; ++r)
              CK(cudaMemcpyAsync((char*)dRecv + ((size_t)r * cnt + (size_t)k * scnt) * 4,
                                 (char*)dTmp + (size_t)r * scnt * 4, scnt * 4,
                                 cudaMemcpyDeviceToDevice, s));
          }
        };
        CK(cudaMemsetAsync(dRecv, 0, cnt * 2 * 4, s));
        issue(); CK(cudaStreamSynchronize(s));
        bool ok = true;
        if (mode) {
          CK(cudaMemcpy(hd.data(), dRecv, cnt * 2 * 4, cudaMemcpyDeviceToHost));
          for (int r = 0; r < 2 && ok; ++r) for (size_t i = 0; i < cnt; ++i)
            if (hd[r * cnt + i] != pat(r, (int)i)) { ok = false; break; }
        }
        for (int w = 0; w < 5; ++w) issue();
        const int reps = 30;
        CK(cudaEventRecord(e0, s));
        for (int k = 0; k < reps; ++k) issue();
        CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        double us = ms * 1000.0 / reps;
        printf(", \"split%d_%s\": {\"us\": %.1f, \"GBs\": %.2f, \"ok\": %s}",
               S, mode ? "copy" : "pure", us, (double)cnt * 4 / (us * 1e-6) / 1e9,
               mode ? (ok ? "true" : "false") : "null");
        fflush(stdout);
      }
    }
    cudaFree(dTmp);
  }
  printf("}\n");
  cudaFree(dSend); cudaFree(dRecv);
  ncclCommDestroy(comm);
  close(fd);
  return 0;
}