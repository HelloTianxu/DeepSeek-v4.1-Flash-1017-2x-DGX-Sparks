#!/usr/bin/env python3
"""E-SELF-09：S(cp.async 级数) × MINB(__launch_bounds__ 最小 block 数) 扫描版构建。

与 probe-build.py 同配方，只把 experts_grouped.cuh 换成带模板参数的 sweep 版，
产物名/目录都另起，**不覆盖 E-SELF-07/08 的基线产物**。
一次编译出整个二维网格，运行时用 TF_PROBE_M2_S / TF_PROBE_M2_MINB 选臂 ⇒ 同进程 A/B。
"""
import os, time, pathlib, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="~/ptxwork/exl3src-sweep")
ap.add_argument("--out", default="~/ptxwork/probe-build-sweep")
ap.add_argument("--name", default="tf_exl3_sweep")
ap.add_argument("--jobs", default="5")
args = ap.parse_args()

SRC = pathlib.Path(os.path.expanduser(args.src))
OUT = pathlib.Path(os.path.expanduser(args.out))
OUT.mkdir(parents=True, exist_ok=True)

import torch
from torch.utils.cpp_extension import load

names = ["experts.cpp", "experts.cu", "experts_cb0.cu", "experts_cb1.cu", "experts_cb2.cu"]
sources = [str(SRC / n) for n in names]
for s in sources:
    assert os.path.isfile(s) and os.path.getsize(s) > 0, f"源文件缺失或为空: {s}"
hdr = SRC / "experts_grouped.cuh"
assert hdr.is_file() and hdr.stat().st_size > 0, "头文件缺失"
txt = hdr.read_text()
assert "TF_PROBE_M2_S" in txt, "头文件不是 sweep 版（缺环境变量选臂），拒绝构建"
assert "int S_ = 0, int MINB = 1" in txt, "头文件缺模板参数改动"

print("torch", torch.__version__, flush=True)
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)
print("src", str(SRC), flush=True)

t0 = time.time()
os.environ.setdefault("MAX_JOBS", args.jobs)
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1"
m = load(
    name=args.name,
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
assert "grouped_mma2" in have and "grouped_mma" in have, "缺少目标导出"
so = OUT / f"{args.name}.so"
assert so.is_file() and so.stat().st_size > 0, "产物 .so 缺失或为空"
print("SO_BYTES", so.stat().st_size, flush=True)
print("BUILD_OK", flush=True)