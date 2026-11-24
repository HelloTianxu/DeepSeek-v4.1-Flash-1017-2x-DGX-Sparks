#!/usr/bin/env python3
"""E-UPSTREAM-01：ROUND_ROWS 16→32/48 的 arena 增量测算。

源：真·生产 pin fbbc9713 的 round.zig:200-300（不再是近似树 f77d5a1）。
逐项照抄 a.take(...)，只有 glue 按 max_rows(48) 固定，其余按设定值 R 线性。
已确证常量：hl=64/2=32, attn_splits=8, tile=64, hc_blocks=40,
nb_max=bucketFor(POOL,POOL)=POOL, candidate_block=8, candidate_blocks=2048。
未取到精确值的项（wo_k / ew.k / ew.n / en / slots）单列，不参与主结论。
"""
import json

MIB = 2**20
GIB = 2**30

POOL = 1052672
WORLD = 2
D = 5120
HC = 4
HD = 512
Q_LORA = 1280
HL = 64 // WORLD          # num_attention_heads 64, TP2
IDX_HEADS = 32
IDX_HD = 128
IDX_TOPK = 512
EXPERTS = 384
ATTN_SPLITS = 8           # tri_attn.zig:14
TILE = 64                 # tri_index.zig:14
HC_BLOCKS = 40            # tri_basic.zig:12
CAND_BLOCK = 8            # candidate_block_size
CAND_BLOCKS = 2048        # candidate_topk_blocks
MAX_ROWS = 48             # round.zig:36
BUCKET_MIN = 1024


def bucket_for(n, cap):
    b = BUCKET_MIN
    while b < n:
        b *= 2
    return min(b, max(cap, BUCKET_MIN))


NB_MAX = bucket_for(POOL, POOL)


def round_bytes(R):
    """R 线性项逐条照抄 round.zig。返回 (总字节, 明细dict)。"""
    t = {}
    t["inputs"] = 5 * R * 8
    t["h"] = R * HC * D * 2
    t["h_alt"] = R * HC * D * 2
    t["x"] = R * D * 2
    t["part"] = R * HC_BLOCKS * 32 * 4
    for k in ("pre", "pre_a", "pre_f", "post"):
        t[k] = R * HC * 4
    t["comb"] = R * HC * HC * 4
    t["qa"] = R * Q_LORA * 2
    t["qr"] = R * Q_LORA * 2
    t["xhq"] = 2 * R * Q_LORA * 2
    t["ykv"] = R * HD * 2
    t["ckv"] = R * HD * 4
    t["cgate"] = R * HD * 4
    t["kout"] = R * HD * 2
    t["q"] = R * HL * HD * 2
    t["o"] = R * HL * HD * 2
    t["iq"] = R * IDX_HEADS * IDX_HD * 2
    t["iq4"] = R * IDX_HEADS * IDX_HD * 2
    t["iw"] = R * IDX_HEADS * 2
    t["keys"] = R * NB_MAX * 8
    t["score"] = R * NB_MAX * 4
    t["cand"] = R * (NB_MAX // CAND_BLOCK + 1)
    t["bmax"] = R * (NB_MAX // CAND_BLOCK + 1) * 4
    t["pidx"] = R * CAND_BLOCKS * 8
    t["cblk"] = R * CAND_BLOCKS * 4
    t["cidx"] = R * IDX_TOPK * 8
    t["ktop"] = R * IDX_TOPK * 8
    t["tmax"] = R * (NB_MAX // TILE + 1) * 8
    t["pbufs.tpos"] = R * IDX_TOPK * 8
    t["pbufs.cand"] = R * IDX_TOPK * TILE * 8
    t["pbufs.every"] = R * 8
    t["kvg"] = R * 2 * HD * 4
    t["sg"] = R * 2 * HD * 4
    t["lat2"] = R * HD * 2
    t["lat"] = R * HD * 2
    t["ikp"] = R * IDX_HD * 2
    t["ik"] = R * IDX_HD * 2
    t["pm_"] = R * HL * ATTN_SPLITS * 4
    t["pl_"] = R * HL * ATTN_SPLITS * 4
    t["po_"] = R * HL * ATTN_SPLITS * HD * 4
    t["pa"] = R * D * 4
    t["ga"] = WORLD * R * D * 4
    t["pm"] = R * D * 4
    t["gm"] = WORLD * R * D * 4
    t["gl"] = R * (D // 256) * EXPERTS * 4
    return sum(t.values()), t


out = {"nb_max": NB_MAX, "max_rows_const": MAX_ROWS, "rows": {}}
base, base_t = round_bytes(16)
for R in (16, 32, 48):
    tot, t = round_bytes(R)
    out["rows"][R] = {
        "round_linear_MiB": tot / MIB,
        "delta_vs_16_MiB": (tot - base) / MIB,
        "delta_vs_16_GiB": (tot - base) / GIB,
    }
out["top_items_at_16_MiB"] = {
    k: v / MIB for k, v in sorted(base_t.items(), key=lambda x: -x[1])[:8]
}
out["note"] = (
    "glue 按 max_rows=48 固定分配，不随 R 变，故不计入增量。"
    "未计入：xhwo/xb/u/pick/mw/e_in/ek/ekg/ekv（wo_k、ew.k/n、en、slots 未取到精确值），"
    "均为每行小量，不改变量级结论。"
)
out["production_usable_GiB"] = 4.8 - 2.5
out["verdict"] = {
    "R32_fits_production_headroom": out["rows"][32]["delta_vs_16_GiB"] < (4.8 - 2.5),
    "R48_fits_production_headroom": out["rows"][48]["delta_vs_16_GiB"] < (4.8 - 2.5),
}

print(json.dumps(out, ensure_ascii=False, indent=2))
with open(__file__.replace("roundrows_mem.py", "roundrows-mem.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)