#!/usr/bin/env python3
"""口径审计：从权威 raw 重算当前口径的 Amdahl 天花板与归因状态表。

唯一数据源：runs/forge-20261125-reclean/raw/account-raw.json（E-RECLEAN-01，commit 1582d62）
口径：r2_warm（热态第二请求），tok=10762，与基线 2084.42 tok/s 同形状（cold_prompt.py 28038 字符）。

反事实规则：某块归零 ⇒ 新 span = span − 该块 busy 时长（假设该块可与其余完全解耦）。
这是**上界**，不是预测：真实收益受配平与依赖限制，须同场 A/B 实证。禁止跨块连乘。
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
d = json.load(open(ROOT / "runs/forge-20261125-reclean/raw/account-raw.json"))
w = d["r2_warm"]
TOK = 10762
span, busy, idle = w["span"], w["busy_union"], w["idle"]
B = w["buckets"]

print("=" * 78)
print("当前唯一口径（E-RECLEAN-01 r2_warm）")
print("=" * 78)
print(f"tok={TOK}  span={span:.3f}ms  busy_union={busy:.3f}ms  idle={idle:.3f}ms ({w['idle_pct']:.2f}%)")
print(f"实测吞吐 = {TOK/(span/1000):.2f} tok/s   （权威基线中位 2084.42）")
print(f"busy_sum={w['busy_sum']:.3f}ms  重叠={w['busy_sum']-busy:.3f}ms")
print()
print("%-16s %12s %9s %7s   %s" % ("块", "ms", "份额%", "n", "归因状态"))
STATUS = {
    "专家 exl3 核心": "✅ 已判决：主因=反量化指令数(弹性0.228)，并发次因(0.124)，shared末因(0.044)",
    "注意力/HC": "❌ 从未归因（本轮之外的最大空白）",
    "集合通信": "✅ 已判决：达屋顶92.2%，仅1.085×余量，杠杆收敛到字节数",
    "稠密 GEMM": "◐ 部分：_gemm 62.74 TFLOPS=FP16屋顶62.2%（计算受限）；旧口径份额已废",
    "专家周边": "❌ 从未单独归因",
    "其他": "❌ 从未归因（含 cuBLAS nvjet 0.73%）",
    "稠密 unpack": "◐ 部分：11.0×字节膨胀但摊薄仅5.47GB/s ⇒ 融合上界=自身份额",
}
tot = 0.0
for k, v in sorted(B.items(), key=lambda x: -x[1]):
    tot += v
    print("%-16s %12.3f %8.2f%% %7d   %s" % (k, v, 100 * v / span, w["counts"][k], STATUS.get(k, "?")))
print("%-16s %12.3f %8.2f%%" % ("合计(busy_sum)", tot, 100 * tot / span))
print()
print("=" * 78)
print("当前口径 Amdahl 天花板（反事实上界，非预测）")
print("=" * 78)


def ceil(name, removed):
    ns = span - removed
    print("%-42s 新span %8.3fms ⇒ %8.2f tok/s  (%+.2f%%)" %
          (name, ns, TOK / (ns / 1000), 100 * (TOK / (ns / 1000) / (TOK / (span / 1000)) - 1)))


core = B["专家 exl3 核心"]
edge = B["专家周边"]
attn = B["注意力/HC"]
comm = B["集合通信"]
dense = B["稠密 GEMM"] + B["稠密 unpack"]
ceil("专家核心归零", core)
ceil("专家路径归零(核心+周边)", core + edge)
ceil("专家核心+注意力HC归零", core + attn)
ceil("专家路径+注意力HC归零", core + edge + attn)
ceil("全部GPU工作归零(只剩idle)", busy)
print()
print("对照：旧口径(8628 tok / 4972.654ms 窗口)曾给出 core 归零 2815.83、core+idle 归零 3604.90。")
print("      旧口径两重污染（AIO=1 + token 数差 24.7%），数字不可用于当前判断。")
print()
print("=" * 78)
print("目标 4000 tok/s 的可达性（当前口径）")
print("=" * 78)
need = TOK / 4000 * 1000
print(f"4000 tok/s 需要 span ≤ {need:.3f}ms，当前 {span:.3f}ms ⇒ 须削 {span-need:.3f}ms = {100*(span-need)/span:.2f}%")
print(f"专家核心全部归零只削 {core:.3f}ms ⇒ {'够' if core >= span-need else '不够'}（缺 {max(0,span-need-core):.3f}ms）")
print(f"专家路径全部归零削 {core+edge:.3f}ms ⇒ {'够' if core+edge >= span-need else '不够'}")
print("⇒ 结论：只靠专家路径不足以达 4000；必须同时动注意力/HC 或通信字节数。")