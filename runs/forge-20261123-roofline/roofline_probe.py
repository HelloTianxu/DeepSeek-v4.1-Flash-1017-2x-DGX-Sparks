#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E-ROOF-01 —— 实测 GB10 的 roofline 两个分母，并按真实专家形状定位。

背景：docs/performance/01 §4 明确缺口——"实际DRAM字节、执行FLOP、同格式可持续吞吐均缺失"，
因此专家核心 1.9085s 到底是算力受限还是带宽受限，此前无法判决。
本探针不改模型/内核/配置，只在停服窗口用同镜像临时容器测硬件分母。

测三件事：
  A 可持续 DRAM 读带宽（8GiB 工作集，远大于 24MB L2）
  B FP16 稠密 GEMM 算力上限（大方阵，compute-bound）
  C 真实专家形状的 grouped GEMM（bmm），扫每专家 M，给出 achieved TFLOPS 与 GB/s

口径纪律：
  · 同缓冲重复读在 <=L2 时测的是缓存带宽，不是 DRAM，必须分开标注
  · achieved GB/s 一律标注工作集规模
  · bmm 用 FP16 权重，字节数是 EXL3 2.9bpw 的约 5.5 倍，
    因此 C 的 GB/s 不能直接与生产专家内核比，只能用于形状/摊销趋势
  · 所有结果汇入依赖链并 synchronize，避免惰性求值剪掉工作
"""
import json
import os
import statistics
import sys
import time

import torch

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/roofline.json"
dev = torch.device("cuda")
props = torch.cuda.get_device_properties(0)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bench(fn, reps=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts), min(ts), max(ts)


res = {
    "device": {
        "name": props.name, "sm_count": props.multi_processor_count,
        "compute_capability": f"{props.major}.{props.minor}",
        "L2_bytes": props.L2_cache_size,
        "total_memory_bytes": props.total_memory,
        "torch": torch.__version__,
    },
    "limits": [
        "停服窗口测得，无模型驻留、无其他GPU负载",
        "同缓冲重复读在工作集<=L2时测的是缓存带宽，非DRAM",
        "bmm为FP16权重，字节数远高于EXL3 2.9bpw，GB/s不可与生产内核直接比",
        "gate+up合并为(E,D,2I)、down为(E,I,D)，共三矩阵，与6*D*I的FLOP口径一致",
        "GEMM上限为cuBLAS FP16稠密路径，非EXL3融合解码内核的可达值",
    ],
}
log(f"设备 {props.name} SM{props.major}.{props.minor} "
    f"{props.multi_processor_count}SM L2={props.L2_cache_size/2**20:.0f}MB")

# ---------- A 可持续读带宽 ----------
log("=== A 读带宽（sum，纯读）===")
bw_rows = []
for gib in (8.0, 1.0, 0.25, 0.0625, 0.0234, 0.0078):
    n = int(gib * (2**30) / 2) // 65536 * 65536
    x = torch.randn(n, device=dev, dtype=torch.float16)
    torch.cuda.synchronize()
    nbytes = x.numel() * 2
    iters = max(1, int(8 * (2**30) / nbytes))

    def fn(x=x, iters=iters):
        a = torch.zeros((), device=dev, dtype=torch.float32)
        for _ in range(iters):
            a = a + x.sum(dtype=torch.float32)
        return a

    med, lo, hi = bench(fn, reps=12, warm=3)
    gbs = nbytes * iters / (med / 1000) / 1e9
    bw_rows.append({
        "working_set_MB": nbytes / 1e6, "iters": iters,
        "ms_median": med, "GBs": gbs,
        "cache_resident_possible": nbytes <= props.L2_cache_size,
    })
    log(f"  工作集{nbytes/1e6:9.2f}MB ×{iters:5d}  {med:8.3f}ms  {gbs:7.1f} GB/s"
        f"{'  (<=L2,缓存)' if nbytes <= props.L2_cache_size else ''}")
    del x
    torch.cuda.empty_cache()
res["read_bandwidth"] = bw_rows
dram = [r for r in bw_rows if not r["cache_resident_possible"]]
res["dram_read_GBs"] = max(r["GBs"] for r in dram)
log(f"DRAM 可持续读带宽（>=64MB工作集取最好）= {res['dram_read_GBs']:.1f} GB/s")

# ---------- B FP16 GEMM 算力上限 ----------
log("=== B FP16 稠密 GEMM 上限 ===")
gemm_rows = []
for n in (16384, 8192, 4096):
    a = torch.randn(n, n, device=dev, dtype=torch.float16)
    b = torch.randn(n, n, device=dev, dtype=torch.float16)
    torch.cuda.synchronize()

    def fn(a=a, b=b):
        return a @ b

    med, lo, hi = bench(fn, reps=15, warm=5)
    tf = 2 * n**3 / (med / 1000) / 1e12
    gemm_rows.append({"n": n, "ms_median": med, "TFLOPS": tf})
    log(f"  {n}^3  {med:8.3f}ms  {tf:7.2f} TFLOPS")
    del a, b
    torch.cuda.empty_cache()
res["gemm_fp16"] = gemm_rows
res["peak_fp16_TFLOPS"] = max(r["TFLOPS"] for r in gemm_rows)
res["machine_balance_FLOP_per_byte"] = (res["peak_fp16_TFLOPS"] * 1e12
                                        / (res["dram_read_GBs"] * 1e9))
log(f"FP16 峰值 {res['peak_fp16_TFLOPS']:.2f} TFLOPS，"
    f"机器平衡 {res['machine_balance_FLOP_per_byte']:.1f} FLOP/byte")

# ---------- C 真实专家形状 ----------
# head 本地 I=1152（TP2），D=5120；gate/up 合并在 N 维，down 转置
# 每专家有效 FLOP/token = 6*5120*1152 = 35389440（gate+up+down 合计）
log("=== C 专家形状 grouped GEMM（FP16 bmm）===")
E, D, I = 384, 5120, 1152
shape_rows = []
for M in (2, 4, 8, 16, 32, 64, 128, 256):
    # gate/up: (E, M, D) x (E, D, I)；down: (E, M, I) x (E, I, D)
    # 必须三个矩阵：gate+up 合并为 (E,D,2I)，down 为 (E,I,D)。
    # r1 错误：FLOP 按 6*D*I（三矩阵）算却只分配两矩阵，TFLOPS 虚高 1.5×。
    xg = torch.randn(E, M, D, device=dev, dtype=torch.float16)
    wgu = torch.randn(E, D, 2 * I, device=dev, dtype=torch.float16)
    xd = torch.randn(E, M, I, device=dev, dtype=torch.float16)
    wd = torch.randn(E, I, D, device=dev, dtype=torch.float16)
    torch.cuda.synchronize()
    wbytes = (wgu.numel() + wd.numel()) * 2
    assert wbytes == 3 * E * D * I * 2, wbytes

    def fn(xg=xg, wgu=wgu, xd=xd, wd=wd):
        return torch.bmm(xg, wgu).sum() + torch.bmm(xd, wd).sum()

    med, lo, hi = bench(fn, reps=15, warm=5)
    flop = E * M * 6 * D * I
    tf = flop / (med / 1000) / 1e12
    gbs = wbytes / (med / 1000) / 1e9
    ai = flop / wbytes
    shape_rows.append({
        "M_per_expert": M, "experts": E, "ms_median": med,
        "TFLOPS": tf, "weight_GBs": gbs,
        "weight_bytes_FP16": wbytes, "arithmetic_intensity": ai,
        "bound_by_roofline": "memory" if ai < res["machine_balance_FLOP_per_byte"] else "compute",
    })
    log(f"  M={M:4d}  {med:8.3f}ms  {tf:6.2f} TFLOPS  权重{gbs:7.1f} GB/s  "
        f"AI={ai:6.1f}  {'带宽侧' if ai < res['machine_balance_FLOP_per_byte'] else '算力侧'}")
    del xg, wgu, xd, wd
    torch.cuda.empty_cache()
res["expert_shape_fp16"] = shape_rows

with open(OUT, "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
log(f"-> {OUT}")