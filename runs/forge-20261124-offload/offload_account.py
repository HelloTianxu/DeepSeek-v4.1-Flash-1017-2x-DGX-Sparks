#!/usr/bin/env python3
"""E-OFFLOAD-01：权重卸载 SSD 换 chunk 8192 的可行性测算。

全部输入均为本项目实测或源码读取，无凭空假设。每个数字标注来源。
"""
import json, math

# ---- 实测输入 ----
DRAM_ROOF = 246.2          # GB/s  roofline-r2.json
SSD_SEQ = 4.0              # GB/s  本轮 dd direct 单流 2GiB
SSD_AGG = 5.69             # GB/s  本轮 4 路并发 direct 2GiB
EXPERT_GIB_RANK = 95.1     # GiB   引擎日志 "rank cache: 95.1 GiB ... read whole"
N_LAYER = 40               # config.json num_hidden_layers
N_EXPERT = 384             # config.json n_routed_experts
TOP_K = 6                  # config.json num_experts_per_tok
CORE_MS_LAYER = 20.121639  # ms    layer-shape-ledger R=2048 median_total_core_ms_paired
UNIQ_GB_LAYER = 2.481745926  # GB  layer-shape-ledger R=2048 unique_all_half_GB_mean
ARENA_MIB = 3766.0         # MiB   引擎日志 "arena 3766 MiB used" @cap2048
ARENA_CONST_MIB = 284.0    # MiB   chunk-mem REPORT §2.3 结构上界推导
PROD_FREE = 4.8            # GiB   用户生产基线（head 最小空闲）
MEMGUARD = 2.5             # GiB   config.sh
GIB = 2**30 / 1e9          # GiB -> GB

out = {}

# ---- 1. 专家访问密度：是否存在"不频繁"子集 ----
dens = {}
for cap in (2048, 4096, 8192):
    # 单专家在一个 chunk 内一次都不被路由到的概率（均匀路由假设，真实路由更集中）
    p_cold = (1 - TOP_K / N_EXPERT) ** cap
    dens[cap] = {
        "expected_active": N_EXPERT * (1 - p_cold),
        "p_expert_unused": p_cold,
        "expected_cold_experts": N_EXPERT * p_cold,
    }
out["expert_density"] = dens

# ---- 2. 每 chunk 的专家字节与时间 ----
uniq_gb_chunk = UNIQ_GB_LAYER * N_LAYER
core_s_chunk = CORE_MS_LAYER * N_LAYER / 1000
bw_eff = uniq_gb_chunk / core_s_chunk
out["per_chunk"] = {
    "unique_GB": uniq_gb_chunk,
    "expert_cache_GB": EXPERT_GIB_RANK * GIB,
    "cache_touched_frac": uniq_gb_chunk / (EXPERT_GIB_RANK * GIB),
    "core_s": core_s_chunk,
    "effective_GB_s": bw_eff,
    "dram_roof_util": bw_eff / DRAM_ROOF,
}

# ---- 3. chunk 8192 的 arena 结构缺口 ----
scale_mib = ARENA_MIB - ARENA_CONST_MIB
arena = {}
for cap in (2048, 4096, 8192):
    k = cap / 2048
    total = ARENA_CONST_MIB + k * scale_mib
    arena[cap] = {"arena_MiB": total, "increment_GiB": (total - ARENA_MIB) / 1024}
usable_now = PROD_FREE - MEMGUARD
ctx_half_free = 2.5  # Review §5.4：CONTEXT 524288->262144 释放
out["chunk_memory"] = {
    "arena_by_cap": arena,
    "production_usable_GiB": usable_now,
    "usable_after_ctx_half_GiB": usable_now + ctx_half_free,
    "gap_8192_after_ctx_half_GiB": arena[8192]["increment_GiB"] - (usable_now + ctx_half_free),
}

# ---- 4. 冷权重清单（每 rank，源码 weights.zig 确认加载与切分）----
cold = {
    "embed.weight": {"GiB": 1.233, "sharded": False, "note": "BF16 未量化，两 rank 各存全量；仅行 gather"},
    "head.trellis": {"GiB": 0.462 / 2, "sharded": True, "note": "vocab 切分；prefill 仅末 token 用一次"},
}
cold_total = sum(v["GiB"] for v in cold.values())
out["cold_inventory"] = {
    "per_rank": cold,
    "total_GiB": cold_total,
    "vision_aligner_GiB": 0.0,   # weights.zig 无 vision/aligner 加载点
    "mtp_GiB_excluded": 0.40,    # decode 投机需要，不可卸载
    "gap_coverage": cold_total / out["chunk_memory"]["gap_8192_after_ctx_half_GiB"],
}

# ---- 5. 卸载专家的盈亏平衡（乐观上界：完美重叠 + 字节线性）----
gap_gb = out["chunk_memory"]["gap_8192_after_ctx_half_GiB"] * GIB
dram_t = (uniq_gb_chunk - gap_gb) / bw_eff
ssd_t = gap_gb / SSD_AGG
breakeven_gb = core_s_chunk * SSD_AGG
out["offload_experts"] = {
    "need_offload_GiB": out["chunk_memory"]["gap_8192_after_ctx_half_GiB"],
    "dram_time_s": dram_t,
    "ssd_time_s": ssd_t,
    "new_core_s_perfect_overlap": max(dram_t, ssd_t),
    "core_time_change_pct": 100 * (max(dram_t, ssd_t) / core_s_chunk - 1),
    "breakeven_offload_GiB": breakeven_gb / GIB,
    "ssd_duty_cycle": ssd_t / max(dram_t, ssd_t),
    "ssd_vs_dram_speed_ratio": bw_eff / SSD_AGG,
    "verdict": "净亏" if max(dram_t, ssd_t) > core_s_chunk else "可行",
}

print(json.dumps(out, ensure_ascii=False, indent=2))
with open(__file__.replace("offload_account.py", "offload-account.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)