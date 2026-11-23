#!/usr/bin/env python3
"""Engram 基线之上重算时间账、Amdahl 边界与各杠杆条件投影。
输入全部取自既有权威产物，不新增测量。投影一律标注为反事实。"""
import json

TOK = 8628                      # ledger 第2请求未缓存输入
W_OLD = 4.972653679             # 旧 window（含 Engram 空隙）
GAP   = 0.670695                # 已由 E-ENGRAM-01 消除
CAT = {                         # ledger §3 分类累计（秒）
 "专家core":1.908546, "collective":0.538977, "其他GEMM":0.503544,
 "HC混合":0.398465, "attention":0.258425, "rotation":0.230837,
 "专家epilogue":0.197363, "unpack":0.098744, "其他GPU":0.183340,
 "类别重叠抵消":-0.016293,
}
W_NEW = W_OLD - GAP
ROOF_BW = 246.2                 # GB/s 实测
CORE_BW = 123.1                 # GB/s 唯一字节/时间
out = {"inputs":{"tokens":TOK,"window_old_s":W_OLD,"gap_removed_s":GAP,
                 "window_new_s":W_NEW,"roof_dram_GBs":ROOF_BW}}

def tps(t): return TOK/t
print(f"旧 window {W_OLD:.6f}s -> {tps(W_OLD):.2f} tok/s")
print(f"新 window {W_NEW:.6f}s -> {tps(W_NEW):.2f} tok/s  (Engram 已落地)")
print(f"目标 4000 需 {TOK/4000:.6f}s，缺口 {W_NEW-TOK/4000:.6f}s "
      f"= 现窗口的 {(W_NEW-TOK/4000)/W_NEW*100:.1f}%\n")

print("=== 新基线分类占比 ===")
for k,v in sorted(CAT.items(), key=lambda x:-x[1]):
    print(f"  {k:14} {v:.6f}s  {v/W_NEW*100:5.2f}%")

noncore = W_NEW - CAT["专家core"]
print(f"\n=== Amdahl 反事实边界（新基线，其他项不变）===")
bounds = {
 "专家core 归零": W_NEW-CAT["专家core"],
 "专家core 提到屋顶(2.00x)": W_NEW-CAT["专家core"]*(1-CORE_BW/ROOF_BW),
 "专家core+collective 双归零": W_NEW-CAT["专家core"]-CAT["collective"],
}
for k,t in bounds.items():
    print(f"  {k:28} -> {t:.6f}s = {tps(t):7.2f} tok/s")
out["amdahl_new"]={k:{"seconds":t,"tok_s":tps(t)} for k,t in bounds.items()}
out["noncore_floor_s"]=noncore; out["noncore_ceiling_tok_s"]=tps(noncore)

print(f"\n非专家部分合计 {noncore:.6f}s ⇒ 即使专家免费也只有 {tps(noncore):.2f} tok/s")

# --- chunk 杠杆：用实测 M 曲线的时间比，不连乘 ---
M_TIME={32:60.426,64:60.796,128:66.012}   # roofline-r2 实测 ms
BULK_MS=20.162; TAIL_MS=9.7333; REPLAY_MS=4.9725
print(f"\n=== chunk 条件投影（其他分类不变；非实测）===")
proj={}
for cap in (2048,4096,8192):
    M=cap*6//384
    nb=TOK//cap; tail=TOK%cap
    ratio=M_TIME.get(M,M_TIME[128])/M_TIME[32] if M in M_TIME else None
    if ratio is None: continue
    bulk=nb*20*BULK_MS*ratio/1000
    tailc=20*TAIL_MS/1000 if tail else 0
    rep=20*REPLAY_MS/1000
    core=bulk+tailc+rep
    w=W_NEW-(CAT["专家core"]-core)
    proj[cap]={"M_per_expert":M,"bulk_calls":nb*20,"core_s":core,"window_s":w,"tok_s":tps(w)}
    print(f"  chunk{cap:5} M={M:3} bulk{nb*20:3}次 core={core:.4f}s "
          f"window={w:.4f}s -> {tps(w):7.2f} tok/s  ({tps(w)/tps(W_NEW)-1:+.1%})")
out["chunk_projection"]=proj

# --- 权重流量：chunk 的硬地板 ---
GB_LAYER=2.482
print(f"\n=== 专家权重 DRAM 硬地板（每 rank 每层 {GB_LAYER} GB）===")
for cap in (2048,4096,8192):
    nb=TOK//cap+(1 if TOK%cap else 0)
    gb=nb*20*GB_LAYER
    print(f"  chunk{cap:5}: {nb} chunk × 20 层 = {gb:6.1f} GB "
          f"→ 屋顶最短 {gb/ROOF_BW*1000:6.1f} ms")

# --- collective 纯搬运估算 ---
print(f"\n=== collective 纯搬运 vs 实测 ===")
msg=2048*5120*2/1e6   # MB per all-reduce: chunk_rows x D x bf16（估算，字节未实测）
for rails in (1,2):
    bw=25*rails
    # ring all-reduce 2 ranks: 线上字节 ≈ 1× 消息大小
    per=msg/bw   # MB / (GB/s) = ms
    print(f"  单条 all-reduce {msg:.2f} MB, {rails} rail({bw} GB/s): {per:.3f} ms")
    for n in (160,200,320):
        print(f"    ×{n} 次 = {per*n:.1f} ms  vs 实测 539.0 ms "
              f"⇒ 差 {539.0-per*n:.1f} ms 可能是自旋/同步")
json.dump(out,open("recompute.json","w"),ensure_ascii=False,indent=2)
print("\n写入 recompute.json")
