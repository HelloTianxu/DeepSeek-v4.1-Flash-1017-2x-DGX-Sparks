#!/usr/bin/env python3
"""E-SELF-07：离线单层专家内核回放 harness 的构建脚本。

只编译 kit 公开源码的 5 个文件（experts.py:129 的配方），不加载整模型。
目的：在内存安全的前提下拿到 grouped_mma_kernel / grouped_rows_kernel 的
逐内核 warp stall reason —— 这是从 SM 消费者往上游回溯的唯一读数。

配方来源（读自源码，非猜测）：
  experts.py:129  load(sources=[experts.cpp, experts.cu, experts_cb{0,1,2}.cu],
                       extra_cuda_cflags=["-O3","-lineinfo"])
  build.py:33     -gencode=arch=compute_{maj}{min},code=sm_{maj}{min}
"""
import os, sys, time, pathlib

SRC = pathlib.Path(os.path.expanduser("~/ptxwork/exl3src"))
OUT = pathlib.Path(os.path.expanduser("~/ptxwork/probe-build"))
OUT.mkdir(parents=True, exist_ok=True)

import torch
from torch.utils.cpp_extension import load

names = ["experts.cpp", "experts.cu", "experts_cb0.cu", "experts_cb1.cu", "experts_cb2.cu"]
sources = [str(SRC / n) for n in names]
for s in sources:
    assert os.path.isfile(s) and os.path.getsize(s) > 0, f"源文件缺失或为空: {s}"

print("torch", torch.__version__, flush=True)
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)
print("sources", names, flush=True)

t0 = time.time()
os.environ.setdefault("MAX_JOBS", "5")
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1"
m = load(
    name="tf_exl3_probe",
    sources=sources,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121,code=sm_121"],
    build_directory=str(OUT),
    verbose=True,
)
dt = time.time() - t0
print(f"BUILD_SECONDS {dt:.1f}", flush=True)

have = [n for n in dir(m) if not n.startswith("_")]
print("EXPORTS", len(have), flush=True)
for n in sorted(have):
    print("  ", n, flush=True)
assert "grouped_mma" in have and "grouped_rows" in have, "缺少目标内核导出"
assert "group_place" in have and "group_count" in have, "缺少工作列表构造导出"
print("BUILD_OK", flush=True)