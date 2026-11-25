#!/usr/bin/env python3
"""E-ALG-02 机制探针：只减 cb_pair 解码调用，判决「指令数是否为因」。

⚠️ 这是机制探针，输出数值错误，**绝不可上生产**。目的只有一个：
   在 HMMA / ldmatrix / global 读 / shared 全不变的前提下，纯粹减少 ALU 指令，
   测「时间变化% ÷ 指令变化%」的弹性，判决发射带宽是否真的是限制。

背景：此前所有实测都在「增指令⇒变慢」方向（KB↓/S↓/MINB↑），
      那些改动同时动了 shared/REG/复用模式，不能反推「减指令⇒变快」。
      恒等式 T=N×L/(warp×CLK) 是代数恒等式（L 由 warp%/issue% 反解），不构成因果证据。

档位（编译期宏 TF_DECODE_HALF）：
  0 = 原版
  1 = cb_pair 4次→2次，asm volatile 强制保留全部 8 个 window ⇒ 只减 cb_pair 的 IMAD/IDP4A/PRMT/HFMA2
  2 = cb_pair 4次→2次，不保留 unused window ⇒ 连带减 SHF/LOP3/SHFL

用法：python3 patch-decode-half.py <src_dir>
幂等：已打过则跳过。
"""
import pathlib, sys

root = pathlib.Path(sys.argv[1])
p = root / "experts_grouped.cuh"
t = p.read_text()

MARK = "TF_DECODE_HALF_APPLIED"
if MARK in t:
    print("ALREADY_APPLIED")
    raise SystemExit(0)

old = """    b0[0] = cb_pair<CB>(st[0], st[1]);
    b0[1] = cb_pair<CB>(st[2], st[3]);
    b1[0] = cb_pair<CB>(st[4], st[5]);
    b1[1] = cb_pair<CB>(st[6], st[7]);
}"""
assert t.count(old) == 1, f"锚点命中 {t.count(old)} 次，须唯一"

new = """#if TF_DECODE_HALF >= 1
    // E-ALG-02 机制探针：解码调用减半，另一半复用。数值错误，仅用于测指令弹性。
    b0[0] = cb_pair<CB>(st[0], st[1]);
    b0[1] = b0[0];
    b1[0] = cb_pair<CB>(st[4], st[5]);
    b1[1] = b1[0];
#if TF_DECODE_HALF == 1
    // 强制 8 个 window 全部保持活跃 ⇒ tile_windows 的计算不被 DCE 删除，
    // 从而唯一变量是 cb_pair 的指令数（asm 本身不产生指令）。
    asm volatile("" :: "r"(st[2]), "r"(st[3]), "r"(st[6]), "r"(st[7]));
#endif
#else
    b0[0] = cb_pair<CB>(st[0], st[1]);
    b0[1] = cb_pair<CB>(st[2], st[3]);
    b1[0] = cb_pair<CB>(st[4], st[5]);
    b1[1] = cb_pair<CB>(st[6], st[7]);
#endif
}"""
t = t.replace(old, new)

guard = """#ifndef TF_DECODE_HALF
#define TF_DECODE_HALF 0
#endif
"""
t = guard + t
p.write_text(t)
print(f"PATCH_OK {MARK}")