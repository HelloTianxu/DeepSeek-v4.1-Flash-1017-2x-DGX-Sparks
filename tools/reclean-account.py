#!/usr/bin/env python3
"""E-RECLEAN-01：从当前配置的 nsys sqlite 重建 timeline 并重算全局份额账。

为什么重算：1123 trace 有两重口径污染 —— ① ENGRAM_AIO=1（11-24 已改 0）
② prompt 8628 token（当前权威 10762）。故 E-PLAN-01 的份额不可换算。

本次采集：PREPARE=0 + rank0 nsys cuda-sw，热态 r2 prefill 2084.72 tok/s
（权威基线 2084.42，+0.014%）⇒ trace 开销可忽略，份额可信。

输出 timeline.json 与 1123 同构：
  kernels: [[start_ms, end_ms, stream, name, gx, gy, gz]]
  api:     [[start_ms, end_ms, name]]
时间零点 = 第一个 kernel 的 start（与 1123 同法）。

用法：python3 reclean-account.py <sqlite> <outdir>
"""
import sqlite3, sys, json, pathlib, collections

DB, OUT = sys.argv[1], pathlib.Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
c = sqlite3.connect(DB)

strings = dict(c.execute("SELECT id, value FROM StringIds"))

krows = c.execute("""
SELECT start, end, streamId, shortName, demangledName, gridX, gridY, gridZ
FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start
""").fetchall()
arows = c.execute("""
SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start
""").fetchall()

t0 = krows[0][0]
ms = lambda ns: (ns - t0) / 1e6

kernels = []
for s, e, st, sn, dn, gx, gy, gz in krows:
    nm = strings.get(sn) or strings.get(dn) or "?"
    kernels.append([ms(s), ms(e), st, nm, gx, gy, gz])
api = [[ms(s), ms(e), strings.get(n, "?")] for s, e, n in arows]

json.dump({"kernels": kernels, "api": api}, open(OUT / "timeline.json", "w"))

# ---- 请求切分：按 GPU 空闲的最大间隔找边界（不依赖固定索引）----
# 合并 busy 区间，再取相邻 busy 段之间的空隙
busy = []
for k in sorted(kernels, key=lambda x: x[0]):
    s, e = k[0], k[1]
    if busy and s <= busy[-1][1]:
        busy[-1][1] = max(busy[-1][1], e)
    else:
        busy.append([s, e])
gaps = [(busy[i][1], busy[i + 1][0], busy[i + 1][0] - busy[i][1]) for i in range(len(busy) - 1)]
big = sorted(gaps, key=lambda g: -g[2])[:8]

print(f"kernel n={len(kernels)}  api n={len(api)}  span={kernels[-1][1]-kernels[0][0]:.3f}ms")
print(f"merged busy 段数={len(busy)}")
print("最大 8 个空隙 (ms):")
for a, b, d in big:
    # 该空隙前最后一个 kernel 的索引
    idx = max(i for i, k in enumerate(kernels) if k[1] <= a + 1e-9)
    print(f"  [{a:.3f}, {b:.3f}] {d:.3f}ms  前 kernel idx={idx}")

# ---- 分类账（按真实内核名归类；名字全部实读自 timeline，不猜）----
# 请求切分依据：_embed_init 每 chunk 一次；idx 0-23 引擎自检、24-1195 是 start.sh smoke test。
# r1 = [1196:8351]（6 chunk，5266ms ⇔ 2041.53 tok/s）
# r2 = [8351:15506]（6 chunk，5158ms ⇔ 2084.72 tok/s，热态，与权威基线 2084.42 差 +0.014%）
SLICES = {"r1_cold": (1196, 8351), "r2_warm": (8351, len(kernels))}

EXPERT_CORE = {"grouped_mma_kernel", "grouped_rows_kernel"}
EXPERT_EDGE = {"rot_in_kernel", "rot_many_kernel", "down_combine_kernel", "gateup_epilogue_kernel"}
DENSE_GEMM = {"_gemm", "glinear_kernel", "linear_kernel"}
DENSE_UNPACK = {"unpack_kernel", "unpack8_kernel"}

def bucket(nm):
    if nm in EXPERT_CORE:
        return "专家 exl3 核心"
    if nm in EXPERT_EDGE:
        return "专家周边"
    if nm.startswith("nccl"):
        return "集合通信"
    if nm in DENSE_GEMM or "nvjet" in nm:
        return "稠密 GEMM"
    if nm in DENSE_UNPACK:
        return "稠密 unpack"
    n = nm.lower()
    if any(s in n for s in ("_hc_", "attn", "rope", "norm", "index_score", "rowmm", "kv_")):
        return "注意力/HC"
    return "其他"

def account(ks, label):
    tot = sum(k[1] - k[0] for k in ks)
    span = ks[-1][1] - ks[0][0]
    # 合并 busy 区间求真实空闲（时间轴，不是时长相加）
    bz = []
    for k in sorted(ks, key=lambda x: x[0]):
        if bz and k[0] <= bz[-1][1]:
            bz[-1][1] = max(bz[-1][1], k[1])
        else:
            bz.append([k[0], k[1]])
    union = sum(b - a for a, b in bz)
    idle = span - union
    agg = collections.Counter(); cnt = collections.Counter()
    for k in ks:
        b = bucket(k[3]); agg[b] += k[1] - k[0]; cnt[b] += 1
    print(f"\n=== {label} ===")
    print(f"  span={span:.3f}ms  busy_union={union:.3f}ms  idle={idle:.3f}ms ({100*idle/span:.2f}%)")
    print(f"  kernel时长相加={tot:.3f}ms（>union 说明有并发重叠）")
    for b, v in agg.most_common():
        print(f"  {b:16s} {v:9.3f}ms  {100*v/span:6.2f}% of span  {100*v/tot:6.2f}% of busy  n={cnt[b]}")
    return {"span": span, "busy_union": union, "idle": idle, "idle_pct": 100*idle/span,
            "busy_sum": tot, "buckets": dict(agg), "counts": dict(cnt)}

res = {"n_kernel": len(kernels), "n_api": len(api),
       "gaps_top": [{"start": a, "end": b, "ms": d} for a, b, d in big]}
for label, (i, j) in SLICES.items():
    res[label] = account(kernels[i:j], f"{label}  kernels[{i}:{j}]")
json.dump(res, open(OUT / "account-raw.json", "w"), ensure_ascii=False, indent=1)
print(f"\nWROTE {OUT}/timeline.json  {OUT}/account-raw.json")