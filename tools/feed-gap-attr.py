#!/usr/bin/env python3
"""E-FEED-01：把 prefill 的 GPU 空隙与 host 侧 HtoD 拷贝分开定价。

用法: feed-gap-attr.py <timeline.json> [--req req2] [--min-ms 1] [--big-ms 100]

判决逻辑（三条，缺一条结论就不成立）：
  1. 空隙内是否真无 CUDA API —— 若空隙由 graph 内未记录的 node 造成，
     全程 cuGraphLaunch 次数应与空隙数同量级；否则「采集伪影」假说被否证。
  2. 巨型 HtoD 期间 GPU 忙不忙 —— 忙 ⇒ host 被隐式节流、这段时间是有效计算，
     不是浪费；闲 ⇒ 拷贝本身慢。表观带宽 (字节/时长) 在前者下是假象，不可当带宽读数。
  3. 字节数须由源码断言反解，不可猜：prompt.zig:1357 `m*row_w == n*ew.k`
     ⇒ upload 字节 = n * ew.k * 2。

注意 API 与空隙的关系必须用「严格包含」判定。用区间重叠会把空隙结束后的
API 误算进空隙内部（本次首版即因此误判，见 REPORT §6）。
"""
import argparse
import json
from collections import Counter

SL = {"req1": (24, 6104), "req2": (6104, 12184), "req3": (12184, 18264)}


def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("timeline")
    ap.add_argument("--req", default="req2", choices=sorted(SL))
    ap.add_argument("--min-ms", type=float, default=1.0)
    ap.add_argument("--big-ms", type=float, default=100.0)
    ap.add_argument("--chunk-rows", type=int, default=2048)
    ap.add_argument("--ew-k", type=int, default=6144, help="engram wkv 的 K 维")
    a = ap.parse_args()

    d = json.load(open(a.timeline))
    K, A = d["kernels"], d["api"]
    lo, hi = SL[a.req]
    k = sorted(K[lo:hi], key=lambda x: x[0])
    t0, t1 = k[0][0], k[-1][1]
    span = (t1 - t0) / 1e6
    mg = merge((x[0], x[1]) for x in k)
    busy = sum(e - s for s, e in mg) / 1e6

    gaps = [(mg[i][1], mg[i + 1][0]) for i in range(len(mg) - 1)
            if mg[i + 1][0] - mg[i][1] > a.min_ms * 1e6]
    gl = []
    for g0, g1 in gaps:
        strict = [x for x in A if x[0] >= g0 and x[1] <= g1]
        bef = [x for x in k if x[1] <= g0][-1]
        aft = [x for x in k if x[0] >= g1][0]
        gl.append({
            "ms": round((g1 - g0) / 1e6, 3),
            "at_ms": round((g0 - t0) / 1e6, 3),
            "strict_api_inside": len(strict),
            "api_names": [x[2] for x in strict],
            "prev_kernel": bef[3][:60], "prev_ms": round((bef[1] - bef[0]) / 1e6, 3),
            "next_kernel": aft[3][:60], "next_ms": round((aft[1] - aft[0]) / 1e6, 3),
        })

    nbytes = a.chunk_rows * a.ew_k * 2
    big = []
    for s, e, n in sorted(A, key=lambda x: x[0]):
        if n != "cuMemcpyHtoDAsync_v2" or (e - s) < a.big_ms * 1e6:
            continue
        if not (t0 <= s <= t1):
            continue
        ov = sum(min(ke, e) - max(ks, s) for ks, ke, *_ in k if ks < e and ke > s)
        big.append({
            "at_ms": round((s - t0) / 1e6, 2), "ms": round((e - s) / 1e6, 3),
            "gpu_busy_pct": round(100.0 * ov / (e - s), 1),
            "apparent_MB_s": round(nbytes / ((e - s) / 1e9) / 1e6, 1),
        })

    mc = [x for x in A if t0 <= x[0] <= t1 and x[2] == "cuMemcpyHtoDAsync_v2"]
    small = [x for x in mc if (x[1] - x[0]) < a.big_ms * 1e6]
    gap_ms = sum(g["ms"] for g in gl)
    res = {
        "req": a.req, "span_ms": round(span, 3), "kernel_busy_ms": round(busy, 3),
        "kernel_streams": dict(Counter(x[2] for x in K)),
        "cuGraphLaunch_total": sum(1 for x in A if x[2] == "cuGraphLaunch"),
        "gaps": gl, "gap_total_ms": round(gap_ms, 3),
        "gap_pct_of_span": round(100 * gap_ms / span, 2),
        "upload_bytes_each": nbytes, "upload_MiB_each": round(nbytes / 2 ** 20, 2),
        "big_uploads": big,
        "big_upload_total_ms": round(sum(x["ms"] for x in big), 3),
        "small_uploads": {"n": len(small),
                          "total_ms": round(sum(x[1] - x[0] for x in small) / 1e6, 3)},
        "amdahl_if_gap_zero": {
            "speedup": round(span / (span - gap_ms), 4),
            "tok_s_from_2084.42": round(2084.42 * span / (span - gap_ms), 1)},
    }
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()