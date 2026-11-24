#!/usr/bin/env python3
"""E-KVOFF-01：KV 卸载 SSD 换 chunk 4096 的可行性测算。

全部公式取自源码（ref f77d5a1，与生产 pin fbbc9713 不同版本，但公式已被
引擎自报日志 532/893 MiB 精确验证）：
  shard.poolBytes: 每源 288*comp_rows + 68*full
    full      = pool/ratio + pad_rows
    comp_rows = pool/world/ratio + pad_rows + trash   (sharded)
  model.zig:308 rope_buf = 4 * rope_rows * (rope_dim/2) * 4
  chunk-mem REPORT §2.3: arena = 常量284MiB + (cap/2048)*(3766-284)MiB
"""
import json

GIB = 2**30
MIB = 2**20

# ---- 实测/配置输入 ----
POOL = 1052672
WORLD = 2
PAD = 2                      # shard.zig:48 pad_rows = 2
COMP_ROW = 288               # head_dim/2 + head_dim/16 = 256+32
IDX_ROW = 68                 # index_head_dim/2 + /32 = 64+4
KV_RATIOS = [2, 2, 2, 1]     # kv_source_layer_ids=[2,8,14,20] 对应 compress_ratios
ROPE_DIM = 64                # qk_rope_head_dim
ROUND_MAX_ROWS = 16
ARENA_MIB = 3766.0           # 引擎自报 @cap2048
ARENA_CONST_MIB = 284.0      # chunk-mem §2.3
DEVICE_GIB = 112279240996 / GIB   # metrics device_memory_bytes 实测
EXPERT_GIB = 95.1            # 引擎日志 rank cache
MEMTOTAL = 121.63
RESERVE = 12.2
PROD_FREE = 4.8              # 用户生产基线
MEMGUARD = 2.5


def planes(pool, world, sharded, trash=0):
    n = 0
    for r in KV_RATIOS:
        full = pool // r + PAD
        comp = (pool // world // r + PAD + trash) if sharded else full
        n += COMP_ROW * comp + IDX_ROW * full
    return n


def rope(window):
    rows = window + ROUND_MAX_ROWS + 1
    return 4 * rows * (ROPE_DIM // 2) * 4


out = {}

# ---- 1. 验证公式：与引擎日志 532 / 893 MiB 对照 ----
sh = planes(POOL, WORLD, True)
fl = planes(POOL, WORLD, False)
out["formula_validation"] = {
    "sharded_MiB_computed": sh / MIB, "sharded_MiB_engine_log": 532,
    "replicated_MiB_computed": fl / MIB, "replicated_MiB_engine_log": 893,
    "match": abs(sh / MIB - 532) < 1 and abs(fl / MIB - 893) < 1,
    "sum_inv_ratio": sum(1 / r for r in KV_RATIOS),
}

# ---- 2. CONTEXT 依赖项枚举 ----
rope_now = rope(524288)
ctx_dep = sh + rope_now
out["context_dependent"] = {
    "kv_planes_MiB": sh / MIB,
    "rope_tables_MiB": rope_now / MIB,
    "total_GiB": ctx_dep / GIB,
    "note": "只有这两项随 CONTEXT/POOL 变化；权重与 arena 不随 CONTEXT 变",
}

# ---- 3. 残差归属：此前被误当成 KV 的 5.24 GiB 是什么 ----
resid = DEVICE_GIB - EXPERT_GIB - ARENA_MIB * MIB / GIB - sh / GIB
out["residual_attribution"] = {
    "device_total_GiB": DEVICE_GIB,
    "minus_expert_GiB": EXPERT_GIB,
    "minus_arena_GiB": ARENA_MIB / 1024,
    "minus_kv_planes_GiB": sh / GIB,
    "residual_GiB": resid,
    "old_claim_KV_GiB": 5.24,
    "actual_ctx_dependent_GiB": ctx_dep / GIB,
    "overstatement_factor": 5.24 / (ctx_dep / GIB),
    "residual_is_mostly": "非专家权重(embed 1.233 replicated / attn~1.5 / head 0.231 / mtp非专家 / norm/gate/hc) + cuBLAS工作区 + CUDA图 + RDMA环 + spill staging 128MiB",
}

# ---- 4. chunk 各档需求与可用 ----
usable = PROD_FREE - MEMGUARD
scale = ARENA_MIB - ARENA_CONST_MIB
arena = {}
for cap in (2048, 4096, 8192):
    tot = ARENA_CONST_MIB + (cap / 2048) * scale
    arena[cap] = {"arena_MiB": tot, "increment_GiB": (tot - ARENA_MIB) / 1024}
need4096 = arena[4096]["increment_GiB"]
out["chunk_demand"] = {"arena_by_cap": arena, "production_usable_GiB": usable,
                       "need_4096_GiB": need4096, "gap_now_GiB": need4096 - usable}

# ---- 5. 各方案能否补上 4096 缺口 ----
scen = {}
# a) 卸载全部 KV 池（理论上限，实际不可行：活跃 KV 必须驻留）
scen["A_卸载全部KV池"] = {"free_GiB": sh / GIB, "avail_GiB": usable + sh / GIB}
# b) CONTEXT 减半 524288->262144（配置级）
p2 = planes(524288, WORLD, True)
r2 = rope(262144)
scen["B_CONTEXT减半"] = {"free_GiB": (sh - p2 + rope_now - r2) / GIB,
                        "avail_GiB": usable + (sh - p2 + rope_now - r2) / GIB}
# c) CONTEXT 砍到 131072
p3 = planes(262144, WORLD, True)
r3 = rope(131072)
scen["C_CONTEXT砍到131072"] = {"free_GiB": (sh - p3 + rope_now - r3) / GIB,
                              "avail_GiB": usable + (sh - p3 + rope_now - r3) / GIB}
# d) c + embed vocab 切分（引擎改动，E-OFFLOAD-01 副产品 0.617 GiB）
scen["D_C加embed切分"] = {"free_GiB": (sh - p3 + rope_now - r3) / GIB + 0.617,
                         "avail_GiB": usable + (sh - p3 + rope_now - r3) / GIB + 0.617}
for k, v in scen.items():
    v["fits_4096"] = v["avail_GiB"] >= need4096
    v["margin_GiB"] = v["avail_GiB"] - need4096
out["scenarios"] = scen

# ---- 6. spill tier 的真实语义 ----
out["spill_note"] = {
    "SPILL_GIB": 64,
    "what_it_is": "kept prompt states（空闲会话的KV）持久化到盘，读回替代重算prefill",
    "what_it_is_not": "不是活跃KV池的分页；活跃池必须设备驻留",
    "already_on": True,
}

print(json.dumps(out, ensure_ascii=False, indent=2))
with open(__file__.replace("kv_account.py", "kv-account.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)