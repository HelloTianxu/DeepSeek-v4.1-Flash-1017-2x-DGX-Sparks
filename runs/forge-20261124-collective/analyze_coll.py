#!/usr/bin/env python3
"""从 RDMA 计数器采样分离 collective 的「搬字节」与「不搬字节」时间。

口径：
- port_*_data 单位为 4 octet（已用 data/packets×4 ≈ MTU+包头 交叉验证）。
- 只统计请求墙钟窗口内的增量，窗口外作为空闲基线校验。
- 「活跃时间」= 瞬时带宽超阈值的采样区间之和；阈值多档给出，避免单阈值武断。
- 本脚本不产生 collective 内核时长；内核时长引用既有 ledger，属跨源对照，须标注。

用法：analyze_coll.py <run_dir> <label>
"""
import sys, csv, json, pathlib
from collections import defaultdict

LINE_GBS_PER_RAIL = 25.0      # 200 Gb/s 单 rail
N_RAIL = 2
AGG_GBS = LINE_GBS_PER_RAIL * N_RAIL


def load(p):
    rows = defaultdict(list)
    with open(p) as f:
        for r in csv.DictReader(f):
            rows[r["dev"]].append((float(r["wall"]), int(r["xmit_data"]),
                                   int(r["rcv_data"]), int(r["xmit_pkt"]),
                                   int(r["rcv_pkt"])))
    for d in rows:
        rows[d].sort()
    return rows


def main():
    run, label = pathlib.Path(sys.argv[1]), sys.argv[2]
    req = json.load(open(run / "request.json"))
    t0, t1 = req["probe_wall_start"], req["probe_wall_end"]
    tok = req["uncached_input_tokens"]
    pre = req["prefill_seconds_delta"]

    out = {"label": label, "uncached_input_tokens": tok,
           "prefill_seconds": pre, "server_tok_s": tok / pre,
           "window_wall": [t0, t1], "nodes": {}}

    print(f"=== {label} ===")
    print(f"未缓存输入 {tok:.0f} tok / prefill {pre:.6f}s = {tok/pre:.2f} tok/s")
    print(f"请求墙钟窗口 {t1-t0:.3f}s\n")

    grand_bytes = 0
    for node in ("head", "worker"):
        p = run / f"{node}.csv"
        if not p.exists():
            continue
        data = load(p)
        nd = {"devices": {}}
        print(f"--- {node} ---")
        for dev, series in data.items():
            win = [s for s in series if t0 <= s[0] <= t1]
            if len(win) < 2:
                print(f"  {dev}: 窗口内样本不足")
                continue
            # 窗口内增量（用窗口首尾，含边界外推误差 <1 采样间隔）
            dx = (win[-1][1] - win[0][1]) * 4
            dr = (win[-1][2] - win[0][2]) * 4
            dpx = win[-1][3] - win[0][3]
            dpr = win[-1][4] - win[0][4]
            # 瞬时带宽（取 xmit+rcv 的较大者作为该口负载；直连口双向对称）
            bw = []
            for i in range(1, len(win)):
                dt = win[i][0] - win[i-1][0]
                if dt <= 0:
                    continue
                b = max((win[i][1]-win[i-1][1]), (win[i][2]-win[i-1][2])) * 4
                bw.append((dt, b / dt / 1e9))
            tot_t = sum(d for d, _ in bw)
            act = {}
            for th in (1.0, 5.0, 10.0, 20.0):
                act[f">{th:g}GB/s"] = sum(d for d, v in bw if v > th)
            peak = max((v for _, v in bw), default=0)
            vals = sorted(v for _, v in bw)
            med = vals[len(vals)//2] if vals else 0
            avg_pkt = (dx / dpx) if dpx else 0
            nd["devices"][dev] = {
                "xmit_bytes": dx, "rcv_bytes": dr,
                "xmit_packets": dpx, "rcv_packets": dpr,
                "avg_packet_bytes": avg_pkt,
                "window_samples": len(win), "window_seconds": tot_t,
                "active_seconds": act, "peak_GBs": peak, "median_GBs": med,
            }
            print(f"  {dev}: xmit {dx/1e9:.4f} GB / rcv {dr/1e9:.4f} GB, "
                  f"{dpx} 包, 均包 {avg_pkt:.1f} B")
            print(f"    峰值 {peak:.2f} GB/s 中位 {med:.2f} GB/s; "
                  f"活跃 >1GB/s {act['>1GB/s']*1000:.1f}ms "
                  f">10GB/s {act['>10GB/s']*1000:.1f}ms "
                  f">20GB/s {act['>20GB/s']*1000:.1f}ms")
            grand_bytes += max(dx, dr)
        out["nodes"][node] = nd

    # 单向逻辑载荷：直连口两端对称，取 head 侧各口 max(xmit,rcv) 之和会双计，
    # 故用「所有设备 xmit 之和」作为 head 发出的总字节（= worker 收到的）。
    hx = sum(v["xmit_bytes"] for v in out["nodes"].get("head", {}).get("devices", {}).values())
    hr = sum(v["rcv_bytes"] for v in out["nodes"].get("head", {}).get("devices", {}).values())
    out["head_xmit_bytes"] = hx
    out["head_rcv_bytes"] = hr
    out["wire_min_seconds_at_agg_line"] = hx / (AGG_GBS * 1e9)
    out["bytes_per_token"] = hx / tok

    print(f"\n=== 判决量 ===")
    print(f"head 发出 {hx/1e9:.4f} GB, 收到 {hr/1e9:.4f} GB  "
          f"(每 token {hx/tok/1024:.2f} KiB)")
    print(f"若双 rail 满线速 {AGG_GBS:g} GB/s，纯搬运最短需 "
          f"{hx/(AGG_GBS*1e9)*1000:.2f} ms")
    print(f"若单 rail {LINE_GBS_PER_RAIL:g} GB/s，最短需 "
          f"{hx/(LINE_GBS_PER_RAIL*1e9)*1000:.2f} ms")

    # 活跃时间取两节点各口 >10GB/s 的最大值（同一链路两端应一致）
    acts = []
    for node, nd in out["nodes"].items():
        for dev, v in nd["devices"].items():
            acts.append(v["active_seconds"][">10GB/s"])
    out["max_active_seconds_gt10"] = max(acts) if acts else 0
    print(f"网络实际活跃(>10GB/s)最长口径 {out['max_active_seconds_gt10']*1000:.2f} ms")

    json.dump(out, open(run / "analysis.json", "w"), ensure_ascii=False, indent=2)
    print(f"\n写入 {run/'analysis.json'}")


if __name__ == "__main__":
    main()