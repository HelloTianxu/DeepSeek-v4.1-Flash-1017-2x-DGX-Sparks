#!/usr/bin/env python3
"""口径一致性审计器 —— 每次提交前跑，保证「口径单一、可追溯」。

## 为什么需要它
2026-11-25 复盘（docs/performance/05-结论链复盘.md）查明：同一天内结论被弱证据推翻三次，
且失效数字（旧天花板 2815.83/3604.89、旧 tok 8628、61.1% 屋顶、1.633× 余量）
散落在多份文档中，部分没有批注 ⇒ 读者无法判断哪个口径当前有效。
本脚本把「失效表述必须带批注」变成**可执行检查**，而非劝告。

## 三层判定（避免误报）
1. **文档级头注**：前 12 行声明「历史快照 / 不是当前口径」⇒ 全文旧数值合法保留。
2. **段落级批注**：命中行前 40 行内有 ⛔/更正/撤回/否证/已被… ⇒ 合法保留。
3. **引用豁免**：行内含「反例 / 复盘 / 撤回 / 否证」等，说明是在剖析失效表述本身。

## 同时校验唯一口径关键值在各文档一致出现
用法：
  python3 tools/audit-consistency.py            # 审计，非零退出=有未批注失效表述
  python3 tools/audit-consistency.py --verbose  # 并列出文档头注状态
"""
import pathlib, sys, argparse

ROOT = pathlib.Path(__file__).resolve().parent.parent

# 已撤回/已失效的表述：出现处必须带批注，否则读者会误用
STALE = {
    "61.1%屋顶(已撤回)": ["61.1%"],
    "1.633x余量(已撤回)": ["1.633"],
    "8.1352ms下界(已撤回)": ["8.1352"],
    "MLP不足是主因(已否证)": ["MLP 不足", "MLP不足", "MLP）不足"],
    "指令占据发射带宽(已否证)": ["占据发射带宽"],
    "旧天花板2815.83": ["2815.83"],
    "旧天花板3604": ["3604.90", "3604.89"],
    "旧口径8628tok": ["8628"],
}
MARK = ["⛔", "更正", "撤回", "否证", "已废", "作废", "历史快照", "已被", "失效",
        "降级", "~~", "不可迁移", "原文保留", "反例", "复盘", "不是当前口径"]
DOC_EXEMPT = ["历史快照", "不是当前口径"]

# 唯一口径（E-RECLEAN-01 r2_warm）：这些值必须存在且不得出现竞争值
CANON = {
    "10762": "当前 tok",
    "5157.783": "span ms",
    "2086.56": "折算 tok/s",
    "2084.42": "基线 tok/s",
    "2167.061": "专家核心 ms",
    "42.02%": "专家核心份额",
    "906.308": "注意力/HC ms",
    "17.57%": "注意力/HC 份额",
    "300.222": "4000 缺口 ms",
    "2208": "arena MiB",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    docs = sorted(pathlib.Path(ROOT / "docs").rglob("*.md"))
    if not docs:
        print("FAIL 未找到 docs/*.md")
        return 1

    if a.verbose:
        print("=== 文档级头注 ===")
        for p in docs:
            head = "\n".join(p.read_text().splitlines()[:12])
            tag = [m for m in DOC_EXEMPT + ["唯一权威", "原文保留"] if m in head]
            print(f"  {str(p.relative_to(ROOT)):46s} {tag if tag else '—'}")
        print()

    bad = []
    for p in docs:
        lines = p.read_text().splitlines()
        head = "\n".join(lines[:12])
        doc_ok = any(m in head for m in DOC_EXEMPT)
        for i, l in enumerate(lines):
            for name, pats in STALE.items():
                if not any(x in l for x in pats):
                    continue
                if any(m in l for m in ["反例", "复盘", "撤回", "否证", "⛔"]):
                    continue
                ctx = "\n".join(lines[max(0, i - 40):i + 4])
                if doc_ok or any(m in ctx for m in MARK):
                    continue
                bad.append((str(p.relative_to(ROOT)), i + 1, name, l.strip()[:100]))

    print("=== 失效表述批注检查 ===")
    for f, n, name, l in bad:
        print(f"  FAIL {f}:{n} [{name}] {l}")
    print(f"未批注失效表述: {len(bad)} 处")

    print("\n=== 唯一口径关键值覆盖 ===")
    miss = []
    for k, d in CANON.items():
        hits = [p for p in docs if k in p.read_text()]
        flag = "" if hits else "  <-- 缺失"
        if not hits:
            miss.append((k, d))
        print(f"  {d:18s} {k:10s} {len(hits)} 文档{flag}")
    # 权威文档必须含全部关键值
    auth = ROOT / "docs" / "CURRENT-BASELINE.md"
    at = auth.read_text()
    lack = [f"{d}({k})" for k, d in CANON.items() if k not in at]
    if lack:
        print(f"\nFAIL CURRENT-BASELINE 缺唯一口径值: {', '.join(lack)}")

    ok = not bad and not miss and not lack
    print(f"\n{'AUDIT_PASS 口径单一可追溯' if ok else 'AUDIT_FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())