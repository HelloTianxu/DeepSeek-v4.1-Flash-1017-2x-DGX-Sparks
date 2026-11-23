#!/usr/bin/env python3
"""Decode 每 token / 每轮的激活字节账（从 checkpoint 真实张量尺寸，TP2 切分）。"""
import json, struct, collections, re, math

M = "/home/ztxjzh/.cache/huggingface/tensorfold-dsv41/aee640e2-ablit-0bdc2036/model"
idx = json.load(open(M + "/model.safetensors.index.json"))["weight_map"]
hdr = {}
def head(sh):
    if sh not in hdr:
        with open(M + "/" + sh, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr[sh] = json.loads(f.read(n))
    return hdr[sh]

GIB = 2**30
# 分类累计（排除 vision/aligner：weights.zig 无加载点）
groups = collections.Counter()
per_layer_routed = collections.Counter()   # layer -> bytes
n_expert_per_layer = collections.Counter()
for k, sh in idx.items():
    if k.startswith("vision.") or k.startswith("aligner."):
        continue
    h = head(sh)[k]
    sz = h["data_offsets"][1] - h["data_offsets"][0]
    m = re.match(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.", k)
    if m:
        lay, exp = int(m.group(1)), int(m.group(2))
        per_layer_routed[lay] += sz
        groups["routed_experts"] += sz
        continue
    if k.startswith("mtp."):
        groups["mtp"] += sz
    elif "shared_experts" in k:
        groups["shared_expert"] += sz
    elif k.startswith("embed."):
        groups["embed"] += sz
    elif k.startswith("head."):
        groups["head"] += sz
    else:
        groups["attn_dense_etc"] += sz

# 每层 routed 专家数
for k in idx:
    m = re.match(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.", k)
    if m:
        n_expert_per_layer[int(m.group(1))] += 0
exp_ids = collections.defaultdict(set)
for k in idx:
    m = re.match(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.", k)
    if m:
        exp_ids[int(m.group(1))].add(int(m.group(2)))

n_layer = len(per_layer_routed)
bytes_per_layer_all_exp = sum(per_layer_routed.values()) / n_layer
n_exp = len(exp_ids[min(exp_ids)])
bytes_per_exp_full = bytes_per_layer_all_exp / n_exp
bytes_per_exp_rank = bytes_per_exp_full / 2      # TP2 中间维切分

TOPK = 6
DRAM_ROOF = 246.2e9   # B/s 实测

out = {
    "n_moe_layer": n_layer,
    "n_expert_per_layer": n_exp,
    "bytes_per_expert_full_MB": bytes_per_exp_full / 1e6,
    "bytes_per_expert_rank_MB": bytes_per_exp_rank / 1e6,
    "groups_GiB_full_model": {k: v / GIB for k, v in groups.items()},
}

# 每 rank 常驻非专家（近似：attn/dense 半分，embed replicated，head 半分，shared 半分）
nonexpert_rank = (
    groups["attn_dense_etc"] / 2
    + groups["embed"]
    + groups["head"] / 2
    + groups["shared_expert"] / 2
)
out["nonexpert_resident_rank_GiB"] = nonexpert_rank / GIB

# 单 token decode：每层 6 个 routed + 1 shared
exp_1tok = n_layer * TOPK * bytes_per_exp_rank
shared_1tok = groups["shared_expert"] / 2
attn_1tok = groups["attn_dense_etc"] / 2
head_1tok = groups["head"] / 2
total_1tok = exp_1tok + shared_1tok + attn_1tok + head_1tok

out["per_token_bytes_GB"] = {
    "routed_experts": exp_1tok / 1e9,
    "shared_expert": shared_1tok / 1e9,
    "attn_dense": attn_1tok / 1e9,
    "head": head_1tok / 1e9,
    "total": total_1tok / 1e9,
}
out["roofline_1tok"] = {
    "ms_at_dram_roof": total_1tok / DRAM_ROOF * 1000,
    "tok_s_ceiling": DRAM_ROOF / total_1tok,
}

# MTP 轮：验证 1+draft 个 token，专家取并集（独立路由近似）
for ntok in (1, 2, 3, 4):
    uniq = n_exp * (1 - (1 - TOPK / n_exp) ** ntok)
    e = n_layer * uniq * bytes_per_exp_rank
    tot = e + shared_1tok + attn_1tok + head_1tok
    out.setdefault("roofline_by_verify_tokens", {})[ntok] = {
        "uniq_experts_per_layer": uniq,
        "total_GB": tot / 1e9,
        "ms_at_dram_roof": tot / DRAM_ROOF * 1000,
        "rounds_s_ceiling": DRAM_ROOF / tot,
    }

print(json.dumps(out, ensure_ascii=False, indent=2))