#!/usr/bin/env python3
"""MLP 饱和带宽模型 —— ⛔ 已被 E-ROWS-01 实测否证，保留供审计，禁止用于立项。

否证：本模型预测 R32（3 blocks, w=24, k=4）⇒ 175.20 GB/s，**实测 148.44，误差 +18.03%**。
失效原因：用 w·k 作 MLP 代理，但新增 warp 读的是同一批权重（L2 命中 79.49→81.27%、
DRAM 请求数 +0.08%），不产生新的在飞 DRAM 请求 ⇒ 代理量与真实 MLP 脱钩。
**保留价值**：5 点标定数据本身有效，且它记录了「减并发方向」的响应；
失效的是「增并发方向」的外推。参见手册 §7.35/§7.36/§7.37。

原始说明（保留）：

## 为什么需要新模型
`sim-kernel.py` 用 T = N×L/(warp×CLK) 预测源码改动，实测 +29.33% vs 预测 −18.12%，彻底失败。
根因：那是代数恒等式（手册 §7.29），没有内存级并行（MLP）这一项。

## 模型
有效 DRAM 带宽随「在飞请求规模」饱和：

    B(w, k) = ROOF * w*k / (w*k + C)

    w = warps/SM = blocks/SM × (threads/block)/32
    k = KB（每 warp 每步预取的 k tile 数，作为每-warp outstanding 的代理）
    ROOF = 246.2 GB/s（标定可持续读，非官方 273）
    C    = 38.90（由 KB2 三档 occupancy 扫描反解，三点极差 10.03%）

时间下界 T = DRAM字节 / B。生产工作点 (w=16, k=4) ⇒ B=150.49 GB/s = 屋顶 61.1%。

## 标定与验证（5 个独立点，全部位一致探针）
    KB2 b1 (w=8, k=2)  预测  71.75  实测  71.44   +0.43%
    KB2 b2 (w=16,k=2)  预测 111.12  实测 114.46   −2.92%
    KB2 b3 (w=24,k=2)  预测 135.99  实测 133.18   +2.11%
    KB4 b1 (w=8, k=4)  预测 111.12  实测 108.72   +2.20%
    KB4 b2 (w=16,k=4)  预测 153.12  实测 150.49   +1.75%
最大误差 2.92%。**首次同时解释 warp 数与 KB 两个维度**——这是旧模型做不到的。

## 能力边界（手册 §7.25 的延续，必须遵守）
1. 只预测**带宽侧下界**。不预测指令代价。改 tile 尺寸会让 block 数变化，
   进而改变反量化/ldmatrix 总量，那部分代价**必须实测**，不得用本模型外推。
2. 不预测 bank 冲突、spill、栅栏、尾部效应。
3. 全局折算只按「gate/up 份额 × 该内核加速比」单点替换，**禁止比例连乘**（用户纪律）。
4. 任何预测都必须由同场 A/B + lts 计数器实测确认；模型只用于**排序候选**，不用于立项承诺。

## 用法
  python3 sim-mlp.py                 # 标定表 + 候选配置扫描
  python3 sim-mlp.py --blocks 3 --kb 4
"""
import argparse, json

ROOF = 246.2
C = 38.902
CAL = [("KB2 b1", 8, 2, 71.44), ("KB2 b2", 16, 2, 114.46), ("KB2 b3", 24, 2, 133.18),
       ("KB4 b1", 8, 4, 108.72), ("KB4 b2", 16, 4, 150.49)]
# 唯一口径（E-RECLEAN-01 r2_warm）
TOK, SPAN, GU = 10762, 5157.783, 1466.505
PROD_BW = 150.49
SM, WARP_SLOTS, SH_OPTIN = 48, 48, 101376


def bw(w, k):
    return ROOF * w * k / (w * k + C)


def shared_bytes(S, R, KB, bs_buf=2, cols=64):
    """As[S][R][KB*16+8] half + Bs[bs_buf][KB][cols/16][32] uint4 + rows_sh[R] int"""
    return S * R * (KB * 16 + 8) * 2 + bs_buf * KB * (cols // 16) * 32 * 16 + R * 4


def blocks_of(sh, reg, threads=256):
    by_sh = SH_OPTIN // sh if sh else 99
    by_reg = 65536 // (reg * threads) if reg else 99
    return max(1, min(by_sh, by_reg))


def report(w, k, label=""):
    b = bw(w, k)
    sp = b / PROD_BW
    gu = GU / sp
    span = SPAN - (GU - gu)
    t = TOK / (span / 1000)
    base = TOK / (SPAN / 1000)
    return {"label": label, "warps_per_sm": w, "kb": k, "wk": w * k,
            "bw_GBs": round(b, 2), "roof_pct": round(b / ROOF * 100, 1),
            "speedup_vs_prod": round(sp, 4), "gate_up_ms": round(gu, 3),
            "span_ms": round(span, 3), "tok_s": round(t, 2),
            "global_gain_pct": round((t / base - 1) * 100, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=0)
    ap.add_argument("--kb", type=int, default=4)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    print("=== 标定回代（5 点）===")
    mx = 0.0
    for n, w, k, meas in CAL:
        p = bw(w, k)
        e = (p / meas - 1) * 100
        mx = max(mx, abs(e))
        print(f"{n:8s} w={w:2d} k={k} wk={w*k:3d}  预测{p:7.2f}  实测{meas:7.2f}  {e:+6.2f}%")
    print(f"最大误差 {mx:.2f}%   ROOF={ROOF}  C={C}")

    if a.blocks:
        r = report(a.blocks * a.threads // 32, a.kb, f"blocks={a.blocks} KB={a.kb}")
        print("\n=== 指定配置 ===")
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return

    print("\n=== 候选扫描：shared 预算把 wk 锁死了 ===")
    print(f"{'配置':28s} {'shared':>7s} {'blk':>3s} {'w':>3s} {'wk':>4s} {'带宽':>7s} {'屋顶%':>6s} {'全局%':>7s}")
    cands = []
    for S, R, KB, bb, cols, reg in [
        (3, 64, 4, 2, 64, 124),   # 生产
        (3, 64, 2, 2, 64, 124),
        (3, 64, 8, 2, 64, 124),
        (2, 64, 4, 2, 64, 106),
        (3, 32, 4, 2, 64, 80),    # E-ROWS-01 目标
        (3, 32, 4, 1, 64, 80),
        (3, 64, 4, 1, 64, 80),
        (3, 64, 4, 2, 32, 80),
        (3, 64, 4, 1, 32, 80),
    ]:
        sh = shared_bytes(S, R, KB, bb, cols)
        blk = blocks_of(sh, reg)
        w = blk * 8
        r = report(w, KB, f"S{S} R{R} KB{KB} Bs{bb} C{cols}")
        cands.append(dict(r, shared=sh, blocks=blk, S=S, R=R, bs_buf=bb, cols=cols, reg=reg))
        print(f"{r['label']:28s} {sh:7d} {blk:3d} {w:3d} {w*KB:4d} "
              f"{r['bw_GBs']:7.2f} {r['roof_pct']:6.1f} {r['global_gain_pct']:+7.2f}")
    print("\n注：全局% 是**带宽侧上界**，未扣指令代价。改 R/cols 会让 block 数变化，")
    print("    反量化与 ldmatrix 总量随之变化，该代价必须实测（见能力边界 1）。")
    if a.json:
        json.dump({"roof": ROOF, "C": C, "calibration": CAL, "candidates": cands},
                  open(a.json, "w"), ensure_ascii=False, indent=1)
        print(f"WROTE {a.json}")


if __name__ == "__main__":
    main()