#!/usr/bin/env python3
"""E-OCC-02 纯净 occupancy 探针：只改 blocks/SM，指令与数值完全不变。

原理：KB=4 时模板常量 DYN=false，kernel 使用静态 shared 数组，
      `extern __shared__ mma_dyn[]` **完全不被访问**。
      因此启动时多申请动态 shared 只影响 occupancy，不影响任何计算。
      ⇒ 指令数、HMMA、ldmatrix、global 读、shared 流量、DRAM 全部不变，**输出位一致**。

这是与 E-ALG-02（纯净减指令探针）对等的**纯净改并发探针**，
用于判决「并发不足」是否真是因果杠杆，而非伴生现象。

  pad=0     ⇒ shared 44288 ⇒ blocks 2（生产）
  pad=20000 ⇒ shared 64288 ⇒ blocks 1（warp 33.2% → 16.6%）

两臂共用同一模块，仅切环境变量 TF_MMA_DYN_PAD ⇒ 无模块间差异。

用法：python3 patch-dyn-pad.py <src_dir>   （幂等）
"""
import pathlib, sys

root = pathlib.Path(sys.argv[1])
p = root / "experts_grouped.cuh"
t = p.read_text()

MARK = "TF_DYN_PAD_APPLIED"
if MARK in t:
    print("ALREADY_APPLIED")
    raise SystemExit(0)

# 1) helper：每次 launch 读 env（与 patch-red-pad.py 同风格）
anchor_fn = "template <int CB>\nvoid grouped_mma_launch(const GroupedArgs& a, cudaStream_t stream) {"
assert t.count(anchor_fn) == 1, f"launcher 锚点 {t.count(anchor_fn)} 次"
helper = """// E-OCC-02 TF_DYN_PAD_APPLIED: extra dynamic shared to lower occupancy only.
// KB<=4 uses static shared and never touches mma_dyn, so this changes blocks/SM
// without changing a single instruction or any output bit.
static inline size_t tf_dyn_pad() {
    const char* e = getenv("TF_MMA_DYN_PAD");
    return e ? (size_t)atoi(e) : (size_t)0;
}

"""
t = t.replace(anchor_fn, helper + anchor_fn)

# 2) dyn 加上 pad，并让 attribute 无条件设置（pad>0 而 base=0 时也需要）
old = """        const size_t dyn = KB_ > 4 ? sizeof(half) * MMA_STAGES * MMA_ROWS * (KB_ * 16 + 8) +                    \\
                                         sizeof(uint4) * 2 * KB_ * 4 * 32 : 0;                                  \\
        if (dyn) C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn)); \\"""
assert t.count(old) == 1, f"dyn 锚点 {t.count(old)} 次"
new = """        const size_t dyn = (KB_ > 4 ? sizeof(half) * MMA_STAGES * MMA_ROWS * (KB_ * 16 + 8) +                   \\
                                         sizeof(uint4) * 2 * KB_ * 4 * 32 : 0) + tf_dyn_pad();                  \\
        if (dyn) C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn)); \\"""
t = t.replace(old, new)

p.write_text(t)
print(f"PATCH_OK {MARK}")