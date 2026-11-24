#!/usr/bin/env python3
"""E-SELF-07：离线单层专家内核回放。

目的：在内存安全（只造一层权重 ≈2.55 GB，不加载整模型）的前提下，
复刻生产的 grouped_mma_kernel（gate/up）与 grouped_rows_kernel（down），
以便用 ncu 取逐内核 warp stall reason —— 从 SM 消费者往上游回溯的读数。

形状全部由生产 trace 的 grid 反解并验算（见 REPORT §2）：
  gateup_epilogue grid.x=14336=P  down_combine grid=(2048,40,1)⇒R=2048,D=5120
  grouped_mma grid=(1,18,1204)⇒I=1152,n_gu=602   grouped_rows grid=(1,40,820)⇒n_d=820
k2=6 由 checkpoint safetensors 头实测（384 专家 × w1/w2/w3 全部 k2=6，即 3.0 bpw）。

启动实参读自 exl3_experts.zig:286-303 与 experts.cu 的 wrapper，非猜测。
校验靶：满块单次中位 gate/up 13.3761 ms、down 6.6802 ms（trace 内 grid 满格的 80 次）。
"""
import os, sys, json, time, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--routing", default="shared6", choices=["shared6", "routed7", "skew6", "full64"])
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--out", default="")
ap.add_argument("--which", default="both", choices=["both", "mma", "rows"])
ap.add_argument("--ncu-warm", action="store_true", help="只跑少量迭代供 ncu 采集")
ap.add_argument("--conc", type=int, default=1,
                help="并发流数：>1 时同时在多条流上跑同一内核，用于判定卡点是否为「在飞请求数」")
ap.add_argument("--sk", type=int, default=4, help="gate/up 的 K 分裂数（生产=4）")
ap.add_argument("--wk", type=int, default=4, help="gate/up 每 K 链 warp 数（生产=4，改动可能破坏覆盖，仅用于诊断）")
ap.add_argument("--fill", action="store_true",
                help="对照：把每个专家的末组补满 64 个有效 member（重复已有 code）。"
                     "ucount、组数、权重字节全不变，只让原本空转的 warp 变活跃 —— 隔离死 warp 对 barrier 的贡献")
args = ap.parse_args()

import torch
from torch.utils.cpp_extension import load

BUILD = os.path.expanduser("~/ptxwork/probe-build")
SRC = os.path.expanduser("~/ptxwork/exl3src")
m = load(
    name="tf_exl3_probe",
    sources=[os.path.join(SRC, n) for n in
             ["experts.cpp", "experts.cu", "experts_cb0.cu", "experts_cb1.cu", "experts_cb2.cu"]],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121,code=sm_121"],
    build_directory=BUILD, verbose=False,
)

dev = "cuda"
R, SLOTS = 2048, 7
P = R * SLOTS                 # 14336
D, I = 5120, 1152
E = 385                       # 384 routed + shared@384
MAXM = 2048                   # memberStride(R) = ceil(R/32)*32
MAXU = min(P, E)              # 385
K2 = 6
TW = 4 * K2                   # words a tile


def list_len(p, maxu, rows):
    return (p + maxu * (rows - 1)) // rows


NGU = list_len(P, MAXU, 64)   # 602
ND = list_len(P, MAXU, 32)    # 820
assert (NGU, ND) == (602, 820), (NGU, ND)

# ---------------- 权重：每专家每投影 KT*NTILES*TW words ----------------
WORDS_GU = (D // 16) * (I // 16) * TW      # 320*72*24 = 552960
WORDS_D = (I // 16) * (D // 16) * TW       # 72*320*24 = 552960
assert WORDS_GU == WORDS_D
STRIDE_B = WORDS_GU * 4                    # 2211840 B = 2.212 MB


def make_proj():
    # 随机位流：解码 ALU 与访存代价与数值无关，形状/字节量才是被测对象
    buf = torch.randint(-2**31, 2**31 - 1, (E * WORDS_GU,), dtype=torch.int32, device=dev)
    ptrs = (buf.data_ptr() + torch.arange(E, dtype=torch.int64, device=dev) * STRIDE_B).contiguous()
    k2 = torch.full((E,), K2, dtype=torch.int32, device=dev)
    return buf, ptrs, k2


gate_b, gate_p, gate_k2 = make_proj()
up_b, up_p, up_k2 = make_proj()
down_b, down_p, down_k2 = make_proj()

# ---------------- 激活与输出 ----------------
xg = torch.randn(P, D, dtype=torch.float16, device=dev)
xu = torch.randn(P, D, dtype=torch.float16, device=dev)
xd = torch.randn(P, I, dtype=torch.float16, device=dev)
Z = torch.zeros(P * D, dtype=torch.float32, device=dev)   # 覆盖两种输出上界

counts = torch.zeros(E, dtype=torch.int32, device=dev)
uids = torch.zeros(MAXU, dtype=torch.int32, device=dev)
ucount = torch.zeros(1, dtype=torch.int32, device=dev)
members = torch.full((MAXU, MAXM), -1, dtype=torch.int32, device=dev)
work_gu = torch.zeros(NGU * 2, dtype=torch.int32, device=dev)
work_d = torch.zeros(ND * 2, dtype=torch.int32, device=dev)

# ---------------- 路由 ----------------
torch.manual_seed(1234)
if args.routing == "shared6":
    # 6 个互异 routed + 末位共享专家（zig 注释：the shared expert is the last）
    pick6 = torch.argsort(torch.rand(R, 384, device=dev), dim=1)[:, :6].to(torch.int32)
    pick = torch.full((R, SLOTS), 384, dtype=torch.int32, device=dev)
    pick[:, :6] = pick6
elif args.routing == "routed7":
    pick = torch.argsort(torch.rand(R, 384, device=dev), dim=1)[:, :7].to(torch.int32)
elif args.routing == "full64":
    # 零空转对照：224 个专家各恰好 64 行 ⇒ 每个 64 行程序 4 个 warp 全活，
    # 用于分离「barrier stall 里有多少是死 warp 停在栅栏上白等」。
    # 注意：专家数 224 < 385，权重字节随之减少，故只用于比较 stall 结构，不比总时长。
    idx = torch.arange(P, device=dev, dtype=torch.int32) // 64
    pick = idx.view(R, SLOTS)
else:  # skew6：Zipf 偏斜，Gumbel top-k 无放回抽样
    w = torch.log(1.0 / torch.arange(1, 385, device=dev).float() ** 0.9)
    g = -torch.log(-torch.log(torch.rand(R, 384, device=dev)))
    pick6 = torch.argsort(w[None, :] + g, dim=1, descending=True)[:, :6].to(torch.int32)
    pick = torch.full((R, SLOTS), 384, dtype=torch.int32, device=dev)
    pick[:, :6] = pick6
pick = pick.contiguous()

m.group_count(pick, counts, R, SLOTS, E)
m.group_place(pick, counts, uids, ucount, members, R, SLOTS, E)
m.m2_work(counts, uids, ucount, work_gu, 64)
m.m2_work(counts, uids, ucount, work_d, 32)
torch.cuda.synchronize()

nu = int(ucount.item())
cnt = counts[:nu].to(torch.int64)
uid_l = uids[:nu].to(torch.int64)
live_rows = counts[uid_l].to(torch.int64)
g_gu = ((live_rows + 63) // 64)
g_d = ((live_rows + 31) // 32)
live_gu, live_d = int(g_gu.sum()), int(g_d.sum())
pad_gu, pad_d = live_gu * 64, live_d * 32

print(f"routing={args.routing}  ucount={nu}  sum_rows={int(live_rows.sum())} (应={P})")
print(f"  gate/up 活跃组 {live_gu} / 容量 {NGU}   补齐行数 {pad_gu} / 有效 {P} = {pad_gu/P:.3f}x")
print(f"  down    活跃组 {live_d} / 容量 {ND}   补齐行数 {pad_d} / 有效 {P} = {pad_d/P:.3f}x")
top = torch.topk(live_rows, 3)
print(f"  最热专家 rows {top.values.tolist()}  最冷 {int(live_rows.min())}  中位 {int(live_rows.median())}")

# 真正被执行的行数：一个 64 行程序里，warp 按 16 行一组，rows_sh[rg*16]<0 的整组跳过。
# 故执行行数 = 对每个程序取 16*ceil(该组有效行数/16)，不是 64*组数（那是容量口径）。
def executed_rows(counts_list, tile):
    tot = 0
    for n in counts_list:
        ng = (n + tile - 1) // tile
        for g in range(ng):
            in_grp = min(tile, n - g * tile)
            tot += 16 * ((in_grp + 15) // 16)
    return tot


cl = live_rows.tolist()
ex_gu, ex_d = executed_rows(cl, 64), executed_rows(cl, 32)
print(f"  执行行数 gate/up {ex_gu} / 有效 {P} = {ex_gu/P:.3f}x   （容量口径 {pad_gu/P:.3f}x 是上界，非执行量）")
print(f"  执行行数 down    {ex_d} / 有效 {P} = {ex_d/P:.3f}x   （容量口径 {pad_d/P:.3f}x）")

if args.fill:
    mem = members.cpu()
    for u in range(nu):
        n = int(live_rows[u])
        ng = (n + 63) // 64
        target = ng * 64
        codes = mem[u, :n].clone()
        mem[u, :target] = codes.repeat((target + n - 1) // n)[:target]
    members.copy_(mem)
    torch.cuda.synchronize()
    print(f"  [FILL] 已把末组补满：每专家有效 member 变为 {64*((live_rows+63)//64).sum().item()}，"
          f"组数与 ucount 不变")

# ---------------- 启动 ----------------
LO, HI = K2, K2   # range(6,6) → bucket 1 → 内核 <2,10,...>


def run_mma():
    m.grouped_mma(xg, xu, gate_p, up_p, gate_k2, up_k2, uids, ucount, members, Z,
                  2, D, I, P, args.sk, SLOTS, 2, args.wk, LO, HI, 1, work_gu)


def run_rows():
    m.grouped_rows(xd, xd, down_p, down_p, down_k2, down_k2, uids, ucount, members, Z,
                   1, I, D, P, 1, SLOTS, 2, 8, 4, 1, 2, LO, HI, 0, work_d)


TARGET = {"mma": 13.3761, "rows": 6.6802}
res = {"routing": args.routing, "ucount": nu, "live_gu": live_gu, "live_d": live_d,
       "pad_gu": pad_gu / P, "pad_d": pad_d / P, "k2": K2,
       "mem_alloc_GiB": torch.cuda.memory_allocated() / 2**30}

for tag, fn in (("mma", run_mma), ("rows", run_rows)):
    if args.which not in ("both", tag):
        continue
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    n = 2 if args.ncu_warm else args.reps
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    med = ts[len(ts) // 2]
    res[tag] = {"median_ms": med, "min_ms": ts[0], "max_ms": ts[-1],
                "target_ms": TARGET[tag], "ratio": med / TARGET[tag]}
    print(f"{tag:5s} 中位 {med:.4f} ms  min {ts[0]:.4f} max {ts[-1]:.4f}   "
          f"生产靶 {TARGET[tag]:.4f}  比值 {med/TARGET[tag]:.3f}")

print(f"显存占用 {res['mem_alloc_GiB']:.2f} GiB")

if args.conc > 1:
    # 并发判据：同样的内核在 N 条流上同时跑。若卡点是「在飞请求数不足」，
    # 并发会让达成带宽上升、单次等效时长下降；若卡点是访存模式本身，则不变。
    streams = [torch.cuda.Stream() for _ in range(args.conc)]
    for tag, fn in (("mma", run_mma), ("rows", run_rows)):
        if args.which not in ("both", tag):
            continue
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        # 串行基准：同一会话内重测，不复用上面的数
        seq = []
        for _ in range(args.reps):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            for _ in range(args.conc):
                fn()
            b.record(); torch.cuda.synchronize()
            seq.append(a.elapsed_time(b))
        # 并发：事件必须真正 bracket 侧流的工作 —— 侧流 wait 起始事件，
        # 各侧流结束事件再被默认流 wait，否则默认流上的 event 只量到空档（会得出荒谬的加速比）。
        con = []
        for _ in range(args.reps):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize()
            a.record()
            evs = []
            for s in streams:
                s.wait_event(a)
                with torch.cuda.stream(s):
                    fn()
                e = torch.cuda.Event(True)
                e.record(s)
                evs.append(e)
            for e in evs:
                torch.cuda.current_stream().wait_event(e)
            b.record()
            torch.cuda.synchronize()
            con.append(a.elapsed_time(b))
        seq.sort(); con.sort()
        ms, mc = seq[len(seq)//2], con[len(con)//2]
        print(f"[CONC={args.conc}] {tag}: 串行 {ms:.4f} ms  并发 {mc:.4f} ms  "
              f"并发/串行 {mc/ms:.3f}  等效加速 {ms/mc:.3f}x")
        res[f"conc_{tag}"] = {"seq_ms": ms, "conc_ms": mc, "speedup": ms / mc}

if args.out:
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("WROTE", args.out)
print("PROBE_OK")