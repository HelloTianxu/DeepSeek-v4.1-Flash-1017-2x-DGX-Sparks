#!/usr/bin/env python3
"""专家内核 Mock 硬件仿真器（E-SIM-01）。

目的：在不写 CUDA、不停服、不编译的前提下，预测算法改动的收益，并解释
      「为什么单点优化拿到局部改进却看不到全局效果」（多资源配平）。

════════ 模型 ════
一阶资源竞争 + 依赖延迟模型。三层下界取最大：

  T_issue  = N_inst / (SM · SCH · CLK · issue_rate)        发射界
  T_res(k) = demand_k / capacity_k                          各资源界
  T        = max(T_issue, max_k T_res(k))

  issue_rate = min(1, W_sched / L_eff)                      Little's Law
  W_sched    = warps_active% · WARP_SLOTS / SCH
  L_eff      = L0 + Σ_k s_k · u_k                           平均发射间隔(cycle)
  u_k        = 资源 k 的 cycle 占用率（依赖 T ⇒ 迭代求解）

物理含义：warp 不够多时，每个 warp 平均 L 个 cycle 才能发一条指令（依赖/延迟），
发射槽填不满；warp 足够多时发射率饱和到 1，此时真正的资源界接管。
**这正是「配平」：提并发只改 W_sched，若 L 同步变大或某资源界接管，收益就被吃掉。**

════════ 标定 ════
L0 与 s_k 由实测干预点反解，不是拍脑袋。每个标定点的干预有效性都已由 ncu 机制闸确认。
**外推纪律**：模型只在标定点覆盖的范围内可信；超出范围输出会标注 EXTRAPOLATED。

用法：
  python3 tools/sim-kernel.py calibrate      # 读实测点、反解参数、报告拟合误差
  python3 tools/sim-kernel.py scan           # 扫描算法改动空间
  python3 tools/sim-kernel.py explain        # 打印当前瓶颈分解（配平诊断）
"""
import csv, json, math, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ── 硬件常量（实测/官方，见 CURRENT-BASELINE）──
HW = dict(
    SM=48, SCH=4, CLK=2.273e9,          # GB10：48 SM，每 SM 4 scheduler，实测 GPC 2.273 GHz
    WARP_SLOTS=48,                       # 每 SM warp 槽（由 33.19%⇔16warps、16.58%⇔8warps 反证）
    REG_FILE=65536, SHARED=101376,       # 每 SM；shared 实读 config_size 102400
    DRAM_BW=246.2e9,                     # 可持续读实测（官方 273）
    FP16_FLOPS=100.82e12,
    L1TEX_WF_PER_CYCLE=1,                # 每 SM 每 cycle 1 个 shared wavefront
)


def load_ncu(path):
    """ncu CSV：跳过非数字首列的前言行；metric 名在 r[-3]、值在 r[-1]。"""
    d, name = {}, None
    for r in csv.reader(open(path)):
        if len(r) > 3:
            if name is None:
                for c in r:
                    if "grouped_" in c:
                        name = c.split("(")[0].replace("void ", "").strip()
                        break
            try:
                d[r[-3]] = float(r[-1])
            except ValueError:
                pass
    return name, d


def make_elem(name, d, label):
    """把 ncu 读数整理成模型要素。缺失项显式标 None，绝不填默认值冒充实测。"""
    T = d.get("gpu__time_duration.sum", 0) / 1e9
    N = d.get("smsp__inst_executed.sum", 0)
    wp = d.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0)
    wf = d.get("l1tex__data_pipe_lsu_wavefronts_mem_shared.sum")
    miss = d.get("lts__t_sectors_op_read_lookup_miss.sum")
    e = dict(
        label=label, kernel=name, T=T, N_inst=N, warps_pct=wp,
        shared_wf=wf,
        dram_bytes=(miss * 32) if miss is not None else None,
        tensor_pct=d.get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"),
        fmaheavy_pct=d.get("sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active"),
        aluheavy_pct=d.get("sm__pipe_aluheavy_cycles_active.avg.pct_of_peak_sustained_active"),
        REG=d.get("launch__registers_per_thread"),
        shared_block=d.get("launch__shared_mem_per_block_static"),
        lim_reg=d.get("launch__occupancy_limit_registers"),
        lim_sh=d.get("launch__occupancy_limit_shared_mem"),
        spill_st=d.get("l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum"),
        issue_pct=d.get("smsp__issue_active.avg.pct_of_peak_sustained_active"),
    )
    e["cycle_per_sm"] = HW["CLK"] * T
    e["issue_rate"] = N / (HW["SM"] * HW["SCH"] * HW["CLK"] * T) if T else 0
    e["W_sched"] = wp / 100 * HW["WARP_SLOTS"] / HW["SCH"]
    e["L_eff"] = e["W_sched"] / e["issue_rate"] if e["issue_rate"] else 0
    e["u_shared"] = (wf / HW["SM"] / e["cycle_per_sm"]) if wf else None
    e["u_dram"] = (e["dram_bytes"] / HW["DRAM_BW"] / T) if e["dram_bytes"] else None
    e["blocks"] = min(x for x in [e["lim_reg"], e["lim_sh"]] if x) if (e["lim_reg"] or e["lim_sh"]) else None
    return e


def predict(e, L0, s_shared, s_dram, dN=1.0, dW=1.0, dWF=1.0, dDRAM=1.0, dL0=1.0, iters=80):
    """给定改动倍率，迭代求 T。dN=指令数倍率, dW=warp数倍率, dWF=shared wavefront倍率…"""
    N = e["N_inst"] * dN
    W = e["W_sched"] * dW
    wf = (e["shared_wf"] or 0) * dWF
    db = (e["dram_bytes"] or 0) * dDRAM
    T = e["T"]
    for _ in range(iters):
        cyc = HW["CLK"] * T
        u_s = wf / HW["SM"] / cyc if wf else 0.0
        u_d = db / HW["DRAM_BW"] / T if db else 0.0
        L = L0 * dL0 + s_shared * u_s + s_dram * u_d
        rate = min(1.0, W / L) if L > 0 else 1.0
        T_issue = N / (HW["SM"] * HW["SCH"] * HW["CLK"] * rate)
        T_sh = wf / (HW["SM"] * HW["CLK"] * HW["L1TEX_WF_PER_CYCLE"]) if wf else 0.0
        T_dram = db / HW["DRAM_BW"] if db else 0.0
        # tensor / 数学管线界：按实测 cycle 占用率折算的绝对需求，随指令数缩放
        T_pipe = 0.0
        for k in ("tensor_pct", "fmaheavy_pct", "aluheavy_pct"):
            if e.get(k):
                T_pipe = max(T_pipe, e[k] / 100 * e["T"] * dN)
        Tn = max(T_issue, T_sh, T_dram, T_pipe)
        if abs(Tn - T) < 1e-12:
            T = Tn
            break
        T = 0.5 * T + 0.5 * Tn
    binds = {"issue": T_issue, "shared": T_sh, "dram": T_dram, "pipe": T_pipe}
    return T, binds


def collect():
    """收集全部实测点。每个点都标注来源与干预有效性。"""
    pts = {}
    base = ROOT / "runs/forge-20261125-self-consumer/raw"
    for f, lab in [("pipe-mma.csv", "gu_base_probe"), ("pipe-rows.csv", "down_base")]:
        if (base / f).exists():
            n, d = load_ncu(base / f)
            pts[lab] = make_elem(n, d, lab)
    # down 的 PAD 干预（机制闸已确认：冲突 6.016→2.039，wavefront −24.30%）
    bk = ROOT / "runs/forge-20261125-bank-occ/raw"
    for f, lab in [("bk4-p0.csv", "down_pad0"), ("bk4-p1.csv", "down_pad4")]:
        if (bk / f).exists():
            n, d = load_ncu(bk / f)
            pts[lab] = make_elem(n, d, lab)
    # gate/up 的 KB×MINB 四臂
    sim = ROOT / "runs/forge-20261125-sim/raw"
    for c, lab in [("1,4", "gu_kb4_mb1"), ("1,2", "gu_kb2_mb1"), ("3,4", "gu_kb4_mb3"), ("3,2", "gu_kb2_mb3")]:
        f = sim / f"sim-{c}.csv"
        if f.exists():
            n, d = load_ncu(f)
            pts[lab] = make_elem(n, d, lab)
    return pts


def cmd_explain(pts):
    print("=" * 96)
    print("全要素与瓶颈分解（实测，非仿真）")
    print("=" * 96)
    hdr = "%-14s %9s %13s %7s %7s %8s %8s %8s %8s"
    print(hdr % ("点", "T(ms)", "N_inst", "warp%", "issue%", "L(cyc)", "u_shared", "u_dram", "blocks"))
    for k, e in pts.items():
        print(hdr % (k, "%.4f" % (e["T"] * 1e3), format(int(e["N_inst"]), ","),
                     "%.2f" % e["warps_pct"], "%.2f" % (e["issue_rate"] * 100),
                     "%.3f" % e["L_eff"],
                     "%.4f" % e["u_shared"] if e["u_shared"] else "n/a",
                     "%.4f" % e["u_dram"] if e["u_dram"] else "n/a",
                     str(e["blocks"])))
    print()
    print("瓶颈判读：u_* 是该资源占满 cycle 的比例；issue% 是发射槽利用率。")
    print("若所有 u_* 都远低于 1 而 issue% 也低 ⇒ 既不是资源饱和也不是发射饱和 ⇒ 依赖延迟主导（配平）。")


def calibrate(pts):
    """反解 L0 与暴露系数。优先用干预对（同内核、单因素）。"""
    out = {}
    # ── down：PAD 干预给出 s_shared ──
    if all(k in pts for k in ("down_base", "down_pad0", "down_pad4")):
        base, a, b = pts["down_base"], pts["down_pad0"], pts["down_pad4"]
        # PAD 点只采了 bank/wavefront 计数器，缺 N_inst 与 warps ⇒ 用 down_base 的 L 作锚。
        # PAD 已验证位一致且指令数不变(5,482,600 两臂相同)、warps 不变 ⇒ T 变化全部来自 L。
        L0 = base["L_eff"]
        ratio = b["T"] / a["T"]
        dL = L0 * ratio - L0
        du = (b["u_shared"] or 0) - (a["u_shared"] or 0)
        out["down"] = dict(L_anchor=L0, s_shared=(dL / du if du else 0.0),
                           dT_pct=100 * (ratio - 1), dL=dL, du_shared=du,
                           shared_elasticity=(100 * (ratio - 1)) / (100 * du / a["u_shared"]),
                           note="L 锚自 down_base(pipe-rows)；PAD 两臂指令数与 warps 均未变，已机制闸确认")
    # ── gate/up：KB 干预给出指令弹性；MINB 干预给出 W 效应 ──
    if all(k in pts for k in ("gu_kb4_mb1", "gu_kb2_mb1", "gu_kb2_mb3")):
        a, b, c = pts["gu_kb4_mb1"], pts["gu_kb2_mb1"], pts["gu_kb2_mb3"]
        d = pts["gu_kb4_mb3"]
        out["gu"] = dict(
            L_base=a["L_eff"], u_dram_base=a["u_dram"], u_shared_base=a["u_shared"],
            # 纯 KB 效应（MINB=1 固定）
            kb_dN=b["N_inst"] / a["N_inst"], kb_dT=b["T"] / a["T"],
            kb_dW=b["W_sched"] / a["W_sched"], kb_dL=b["L_eff"] / a["L_eff"],
            kb_elasticity=(b["T"] / a["T"] - 1) / (b["N_inst"] / a["N_inst"] - 1),
            # 纯 MINB 效应（KB=2 固定）——含 spill 代价
            mb_dW=c["W_sched"] / b["W_sched"], mb_dT=c["T"] / b["T"],
            mb_dN=c["N_inst"] / b["N_inst"], mb_dL=c["L_eff"] / b["L_eff"],
            # 反事实：若 MINB 不引入额外指令，纯并发应得多少
            mb_pure_dT=(b["L_eff"] / c["L_eff"]) / (c["W_sched"] / b["W_sched"]),
            # MINB 在 KB=4 下（shared 钉 2，warp 没涨）⇒ 纯 spill 代价
            mb_kb4_dN=d["N_inst"] / a["N_inst"], mb_kb4_dT=d["T"] / a["T"],
            mb_kb4_dW=d["W_sched"] / a["W_sched"],
        )
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "explain"
    pts = collect()
    if not pts:
        print("NO_DATA"); return
    if cmd == "explain":
        cmd_explain(pts)
    elif cmd == "calibrate":
        cmd_explain(pts)
        cal = calibrate(pts)
        print()
        print("=" * 96)
        print("标定结果")
        print("=" * 96)
        print(json.dumps(cal, ensure_ascii=False, indent=1))
        out = ROOT / "runs/forge-20261125-sim/raw/calibration.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"points": pts, "calibration": cal}, open(out, "w"), ensure_ascii=False, indent=1)
        print("WROTE", out)
    elif cmd == "scan":
        cal = calibrate(pts)
        cmd_scan(pts, cal)
    else:
        print("unknown cmd:", cmd)




def cmd_scan(pts, cal):
    """扫描算法改动空间。每个方案标注：预测依据、是否外推、被哪个界卡住。"""
    e = pts["gu_kb4_mb1"]
    gu = cal["gu"]
    s_sh = cal["down"]["s_shared"]      # 跨内核借用，标注为假设
    s_dr = 0.0                          # 未标定：DRAM 延迟暴露系数，保守取 0
    # L_base 是**总**平均发射间隔，已含各资源暴露项 ⇒ L0 必须扣除，否则重复计算。
    L0 = gu["L_base"] - s_sh * (e["u_shared"] or 0) - s_dr * (e["u_dram"] or 0)
    print("=" * 100)
    print("gate/up 硬下界（DRAM 决定，与算法无关）")
    print("=" * 100)
    T_dram = e["u_dram"] * e["T"]
    print(f"DRAM 字节 = {e['dram_bytes']/2**30:.3f} GiB，屋顶 246.2 GB/s")
    print(f"T_dram 下界 = {T_dram*1e3:.4f} ms   当前 {e['T']*1e3:.4f} ms   ⇒ 最多快 {e['T']/T_dram:.3f}×（−{100*(1-T_dram/e['T']):.1f}%）")
    print(f"**任何算法改动都不可能低于此界，除非减少 DRAM 字节（换量化格式/提高复用）。**")
    print()
    print("=" * 100)
    print("方案扫描（预测；EXTRAPOLATED 表示超出标定点覆盖范围）")
    print("=" * 100)
    hdr = "%-34s %9s %8s %7s %7s %-10s %s"
    print(hdr % ("方案", "T(ms)", "vs生产", "issue%", "u_dram", "绑定界", "依据/风险"))
    rows = []

    def show(name, note, **kw):
        kw = dict(kw)
        if "dL0" in kw:
            # 实测 dL0 是总 L 的倍率；换算成对 L0 的倍率，保持资源项独立
            tot = gu["L_base"] * kw.pop("dL0")
            kw["dL0"] = (tot - s_sh * (e["u_shared"] or 0) * kw.get("dWF", 1.0)) / L0
        T, binds = predict(e, L0, s_sh, s_dr, **kw)
        binding = max(binds, key=binds.get)
        cyc = HW["CLK"] * T
        ud = (e["dram_bytes"] * kw.get("dDRAM", 1.0)) / HW["DRAM_BW"] / T
        ir = e["N_inst"] * kw.get("dN", 1.0) / (HW["SM"] * HW["SCH"] * HW["CLK"] * T)
        print(hdr % (name, "%.4f" % (T * 1e3), "%+.2f%%" % (100 * (T / e["T"] - 1)),
                     "%.1f" % (ir * 100), "%.3f" % ud, binding, note))
        rows.append(dict(name=name, T_ms=T * 1e3, delta_pct=100 * (T / e["T"] - 1),
                         issue_pct=ir * 100, u_dram=ud, binding=binding, note=note, knobs=kw))

    show("S0 生产基线", "标定点本身", dN=1, dW=1)
    show("S1 KB=2 (MINB=1)", "已实测 +20.55%，模型自校验", dN=gu["kb_dN"], dW=gu["kb_dW"], dL0=gu["kb_dL"])
    show("S2 KB=2+MINB=3 (含spill)", "已实测 −6.18%，模型自校验",
         dN=gu["kb_dN"] * gu["mb_dN"], dW=gu["kb_dW"] * gu["mb_dW"], dL0=gu["kb_dL"] * gu["mb_dL"])
    show("S3 KB=2+MINB=3 无spill", "★反事实：并发收益不被spill吃掉",
         dN=gu["kb_dN"], dW=gu["kb_dW"] * gu["mb_dW"], dL0=gu["kb_dL"])
    show("S4 KB=4+3blocks 无spill", "★最优候选：保KB复用+提并发(S=2,pad=2,REG≤80)",
         dN=1.0, dW=gu["mb_dW"], dL0=1.0)
    show("S5 KB=4+4blocks 无spill", "外推：需REG≤64且shared≤25344，难度高",
         dN=1.0, dW=2.0, dL0=1.0)
    show("S6 解码指令减半(并发不变)", "外推：算法层精简反量化，warp不变",
         dN=0.5, dW=1.0, dL0=1.0)
    show("S7 解码指令减半+3blocks", "★组合：两条杠杆同时动",
         dN=0.5, dW=gu["mb_dW"], dL0=1.0)
    show("S8 shared流量归零", "上界探测：验证shared不是卡口",
         dN=1.0, dW=1.0, dWF=0.0)
    show("S9 DRAM字节减半", "需换更低位宽格式或提高跨层复用",
         dN=1.0, dW=1.0, dDRAM=0.5)
    print()
    print("注：s_shared 借用 down 的标定值（跨内核假设）；s_dram 未标定取 0（保守）。")
    print("    S3/S4/S5/S6/S7/S9 均为 EXTRAPOLATED —— 必须实测验证，不得直接当收益承诺。")
    out = ROOT / "runs/forge-20261125-sim/raw/scan.json"
    json.dump({"T_dram_lower_bound_ms": T_dram * 1e3, "max_speedup": e["T"] / T_dram, "rows": rows},
              open(out, "w"), ensure_ascii=False, indent=1)
    print("WROTE", out)


if __name__ == "__main__":
    main()
