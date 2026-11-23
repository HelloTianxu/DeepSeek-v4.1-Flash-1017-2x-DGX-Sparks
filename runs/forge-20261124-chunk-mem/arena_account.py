#!/usr/bin/env python3
"""B 阶段前置：chunk_rows 变大时 device arena 的内存增量测算。

来源（只读，未改任何代码）：
  prompt.zig  Chunk.init(e, a, cap, tokens)   cap == model.chunk_rows
  exl3_experts.zig Scratch.sizes(rows, slots, d, i, e)
  tri_basic.hc_blocks / tri_index.tile / tri_attn.attn_splits / key_elems / ring_extra
  checkpoint-config.json（真实模型配置）
注意：源码取自 Spark 上 ~/tensorfold-next-f77d5a1（ref f77d5a1），
      与生产 pin 的 TF_REF=fbbc9713 不同版本，故本测算为**估计**，须以启动日志实测复核。
"""
import json, math

MiB = 1024 ** 2
GiB = 1024 ** 3

# ---- 真实配置 ----
d = 5120            # hidden
hc = 4              # hc_mult
hd = 512            # head_dim
heads_total = 64
world = 2           # TP2
hl = heads_total // world          # plan.heads = c.heads / world = 32
groups = 8          # o_groups
o_lora = 1024
q_lora = 1280       # wq_a.n
experts = 385       # 384 routed + 1 shared
slots = 7           # top6 + shared
idx_heads, idx_hd, idx_topk = 32, 128, 512
ih = idx_heads * idx_hd            # 4096
window = 128
cand_block, cand_blocks = 8, 2048
hc_blocks = 40
ring_extra, raw_rows = 16, 64
key_elems = 1 << 24
tile, attn_splits = 64, 8
ex_d, ex_i, ex_e = 5120, 1152, 385   # TP2 本地 I=1152


def promptZ(dd, ii):
    return max(2 * ii + dd // 2, dd)


def listLen(p, maxu, rows):
    return (p + maxu * (rows - 1)) // rows


def scratch(rows):
    p = rows * slots
    maxu = min(p, ex_e)
    ceil128 = ((rows + 127) // 128) * 128
    return (2 * p * ex_d + 2 * p * ex_d + 2 * p * ex_i + 4 * promptZ(ex_d, ex_i) * p
            + 4 + 4 * maxu + 4 + 4 * ex_e + 4 * maxu * ceil128
            + 8 * listLen(p, maxu, 64) + 8 * listLen(p, maxu, 32))


def cap_linear_terms(cap):
    """逐项列出随 cap 线性的分配（字节）。不含 engram 与 grouped-linear（另计）。"""
    t = {}
    t["ids+pos0+pos1"] = 3 * cap * 8
    t["h+h_alt"] = 2 * cap * hc * d * 2
    t["x"] = cap * d * 2
    t["part"] = cap * hc_blocks * 32 * 4
    t["pre/pre_a/pre_f/post"] = 4 * cap * hc * 4
    t["comb"] = cap * hc * hc * 4
    t["qa+qr"] = 2 * cap * q_lora * 2
    t["y"] = cap * hd * 2
    t["q+o"] = 2 * cap * hl * hd * 2
    t["wsrc"] = cap * hd * 2
    t["u"] = cap * groups * o_lora * 2
    t["pa+xf+pm"] = 3 * cap * d * 4
    t["ga+gm"] = 2 * world * cap * d * 4
    t["neg"] = cap * 8
    t["logits"] = cap * experts * 4
    t["pick/wts/pick_vl/wts_vl"] = 4 * cap * slots * 4
    t["kvc+scc"] = 2 * cap * hd * 4
    t["lat+lat2"] = 2 * cap * hd * 2
    t["ik"] = cap * idx_hd * 2
    t["groups+gpos"] = 2 * cap * 8
    t["iq+iq4"] = 2 * cap * ih * 2
    t["wl+iw"] = cap * idx_heads * 4 + cap * idx_heads * 2
    t["vis+vis_blk"] = 2 * cap * 8
    t["cidx+ktop"] = 2 * cap * idx_topk * 8
    t["cidxp"] = cap * (idx_topk + 16) * 8
    t["every"] = cap * 8
    t["pidx"] = cap * cand_blocks * 8
    t["expert_scratch"] = scratch(cap)
    return t


base = 2048
terms = cap_linear_terms(base)
tot = sum(terms.values())
print("=== cap=2048 时随 chunk 线性的分配（不含 engram / grouped-linear / 常量项）===")
for k, v in sorted(terms.items(), key=lambda x: -x[1])[:12]:
    print(f"  {k:28s} {v/MiB:9.1f} MiB")
print(f"  {'合计':28s} {tot/MiB:9.1f} MiB   = {tot/base/1024:.1f} KiB/行")
print()
print(f"引擎自报 arena = 3766 MiB；本测算的 cap 线性部分 = {tot/MiB:.0f} MiB "
      f"({100*tot/(3766*MiB):.1f}% of arena)")
print()
print("=== chunk 变大时 arena 增量（仅 cap 线性部分，下界）===")
per_row = tot / base
for c in (4096, 8192, 16384):
    new = sum(cap_linear_terms(c).values())
    print(f"  chunk={c:6d}: cap线性 {new/MiB:8.1f} MiB, 增量 +{(new-tot)/MiB:8.1f} MiB "
          f"({(new-tot)/GiB:.2f} GiB)")
print()
print("=== 最坏情形上界：假设整个 arena 都随 cap 线性 ===")
for c in (4096, 8192, 16384):
    print(f"  chunk={c:6d}: arena {3766*c/base:8.0f} MiB, 增量 +{3766*(c/base-1):8.0f} MiB "
          f"({3766*(c/base-1)/1024:.2f} GiB)")
print()
print("=== 结构性上界：arena 里几乎全是 cap*X，常量项有限 ===")
const_terms = {
    "keys(ke*8, ke 封顶 2^24)": (1 << 24) * 8,
    "score(ke*4)": (1 << 24) * 4,
    "gate_f(experts*d*4)": experts * d * 4,
    "tmax((ke/tile+cap)*8)": ((1 << 24) // tile + base) * 8,
    "bmax": max(16 * 16384, (1 << 24) // cand_block + base) * 4,
    "gl": 16 * (d // 256) * experts * 4,
    "blas_ws(典型 32MiB)": 32 * MiB,
    "ws.w(max_kn*2, 估 42MiB)": 42 * MiB,
}
const_sum = sum(const_terms.values())
print(f"  可辨认常量项合计 ≈ {const_sum/MiB:.0f} MiB")
frac_hi = (3766 * MiB - const_sum) / (3766 * MiB)
print(f"  ⇒ cap 线性占比上界 ≈ {100*frac_hi:.0f}%（尚含 Round/Caches/drafter 未计）")
print(f"  ⇒ cap 线性占比区间：{100*tot/(3766*MiB):.0f}%（已枚举下界） .. {100*frac_hi:.0f}%（结构上界）")
print()

# ---- 实测内存与 memguard ----
avail_head = 8616960 / 1048576     # GiB, /proc/meminfo MemAvailable 实测
avail_worker = 10409680 / 1048576
floor = 2.5                        # scripts/config.sh MEMGUARD_GIB 默认
print("=== 安全判定（实测 MemAvailable，memguard 地板 %.1f GiB）===" % floor)
print(f"  head avail {avail_head:.2f} GiB → 可用余量 {avail_head-floor:.2f} GiB")
print(f"  worker avail {avail_worker:.2f} GiB → 可用余量 {avail_worker-floor:.2f} GiB")
print("  (head 是约束方；两 rank 各自分配自己的 arena)")
print()
lo_frac = tot / (3766 * MiB)
for c in (4096, 8192):
    d_lo = tot * (c / base - 1) / GiB
    d_hi = 3766 * (c / base - 1) / 1024
    d_struct = 3766 * frac_hi * (c / base - 1) / 1024
    a_lo, a_hi = avail_head - d_hi, avail_head - d_lo
    verdict = "安全" if a_lo > floor else "不安全：最坏情形跌破 memguard 地板"
    print(f"  chunk={c}: 增量 {d_lo:.2f} .. {d_hi:.2f} GiB（结构估计 {d_struct:.2f}）")
    print(f"     head 加载后 avail {a_lo:.2f} .. {a_hi:.2f} GiB "
          f"(高于地板 {a_lo-floor:+.2f} .. {a_hi-floor:+.2f}) → {verdict}")
print()
print("=== POOL 下界：parts=2 时 POOL >= 2*WINDOW，KV 池无法再缩 ===")
print(f"  WINDOW=524288 ⇒ POOL 最小 1048576；当前 1052672 = 1048576 + 4096（正好多一个 chunk 块）")
print("  ⇒ 想靠缩 KV 池给大 chunk 腾内存，必须先降 CONTEXT（属产品能力变更）")
print()
print("=== POOL 整除约束（硬门：cap % (parts*chunk_rows) != 0 → BadPoolSize）===")
pool_now = 1052672
for c in (2048, 4096, 8192, 16384):
    unit = world * c
    ok = pool_now % unit == 0
    print(f"  chunk={c:6d} 需 POOL % {unit:6d} == 0 : 当前 POOL={pool_now} "
          f"{'✓' if ok else '✗ 拒绝启动'}  (余 {pool_now % unit})")
print(f"  候选 POOL=1048576 (2^20): " +
      ", ".join(f"chunk{c}={'✓' if 1048576 % (world*c) == 0 else '✗'}"
                for c in (2048, 4096, 8192, 16384)))