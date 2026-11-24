# E-SELF-08：判决木桶分析 —— 两个独立干预，同一机制

- 日期：2026-11-25
- 触发：用户「嗯，你来进行实验吧。来做实确定我们这个木桶分析是正确的」
- 前置：E-SELF-07（`runs/forge-20261125-self-consumer/`）从消费者 SM 逐环回溯，指认卡口在**片内递货**
- 结论：**木桶分析成立。** 两个独立干预都把达成 DRAM 带宽顶高约 10%，而 **DRAM 字节几乎不变**；
  且两次占用率**一降一升**，带宽却都涨 ⇒ **warp 数不是卡口，DRAM 容量不是卡口，卡口是片内递货。**

---

## 0. 判决表

| | gate/up 行（v1 → mma2） | down 行（v1 → down3） |
|---|---|---|
| 形状 | N=2304, K=5120, P=14336（两臂同形） | **N=5120, K=1152, P=14336（生产形状）** |
| 时长（cuda event，配对均值） | **−9.39%** | **−6.79%** |
| 时长（ncu 单核） | 25.2749 → 23.0861 ms（−8.7%） | 6.5266 → 5.9495 ms（−8.8%） |
| **DRAM miss 字节** | **3.450 → 3.450 GiB（+0.0%）** | **0.826 → 0.832 GiB（+0.7%）** |
| **达成 DRAM 带宽** | **146.6 → 160.5 GB/s（+9.5%）** | **135.8 → 150.1 GB/s（+10.5%）** |
| 占屋顶（246.2 GB/s） | 59.5% → **65.2%** | 55.2% → **61.0%** |
| **达成占用率** | **32.34% → 15.55%（降）** | **15.30% → 29.43%（升）** |
| Tensor pipe | 28.21% → **46.19%** | 29.94% → **45.60%** |
| math_pipe_throttle | 10.3% → **22.9%** | 8.1% → **19.1%** |
| 首因 stall 变化 | barrier 30.2% → 21.7% | **long_scoreboard 33.1% → 18.5%** |

**关键交叉验证：占用率一降一升，带宽却同向上升。** 若卡口是 warp 数，占用率腰斩的 mma2 必须变慢；
它反而快 8.7%。若卡口是 DRAM 容量，两臂 miss 字节相同就不该有时长差；实测差 8.7%/8.8%。
**两条都被排除，剩下的只有「字节在片内被递到消费者手上的速度」。**

---

## 1. 实验设计：什么被钉住，什么被改

**钉住**：同一路由（`shared6`，385 专家、共享专家独占 2048 行）、同一 P=14336、同一 K、
同一权重字节布局（每 tile `TW=4·K2=24` words = 96 B）、同一 codebook cb2、同一 k2=6（3.0 bpw）、
同一进程同一显存、**A/B/A/B 同场交替**（禁止复用历史基线）。

**改**：权重字的暂存路径。
- v1 `grouped_mma_kernel`（`:704`）：激活走 cp.async 环（`MMA_STAGES=3`，`:613/:645`），
  **权重走「全局→寄存器→shared，提前两步」**（`:558` 注释 + `:597`）。
- mma2（`:893`）/ down3（`:1417`）：**激活与权重双双走 cp.async**
  （`:798` 注释「Rows and trellis words come through a ring of S stages of KB k tiles by cp.async
  (one barrier a stage)」，`:847/:852`）。

**防「测到跳过工作的内核」**：
- gate/up 两臂输出同为 f32 Z，做数值接近性校验：`max|v1|=402.9407`、`max|v1−mma2|=0.002869`、
  **相对 7.119e-06**。源码 `:800` 明示 mma2 与 v1 **非位一致**（split order 不同），故只断言接近而非相等。
- down3 输出 bf16 `Y[P,N]` 且已融合 epilogue，与 v1 的未加权 Z **语义不同**，
  故只做非空 + 有限性断言（`max|Z|=188.4112`、`max|Y|=14.0000`），**不做数值等价断言**。

---

## 2. 机制读数：指令流层面的直接证据

| 指令 | gate/up v1 | gate/up mma2 | down v1 | down down3 |
|---|---:|---:|---:|---:|
| 总 warp 指令 | 4,570,911,360 | **2,758,024,926** | 926,898,040 | **826,754,360** |
| `global_ld` | 40,529,664 | **547,776** | 27,375,400 | **4,384,000** |
| `shared_st` | 38,398,464 | **22,464** | 5,482,600 | **1,865,600** |
| `ldsm`(ldmatrix) | 49,121,280 | 76,677,120 | **0** | **19,169,280** |
| L2 读请求 | 18.146 GiB | **10.567 GiB** | 2.379 GiB | 2.699 GiB |

`global_ld` 掉 74×、`shared_st` 掉 1700× —— **这正是「不再用寄存器搬权重」的指纹**：
权重不再经寄存器落地，故不再需要逐字 global load + 逐字 shared store，改由 cp.async 直入 shared，
再用 ldmatrix 取出（`ldsm` 上升）。

> **零计数器陷阱复发提醒**：`l1tex__t_sectors_pipe_lsu_mem_global_op_ldgsts_cache_access.sum`
> 在**四个内核上全部报 0**，包括源码确证使用 cp.async 的那些。**该计数器在 GB10 不可用**，
> 不可据其断言「无 cp.async」（E-DRAM-01 已记录，手册 §7.10）。

---

## 3. 寄存器双角色：已分离，且主次与直觉相反

E-SELF-07 遗留问题：寄存器既**限制驻留 warp 数**、又**承载权重暂存**，哪个主导？

| | REG/thread | occupancy_limit_registers | 占用率 | 时长 |
|---|---:|---:|---:|---:|
| gate/up v1 | 106 | 2 blocks | 32.34% | 25.27 ms |
| gate/up mma2 | **133（更多）** | **1 block（更少）** | **15.55%** | **23.09 ms** |
| down v1 | 255 | 2 blocks | 15.30% | 6.53 ms |
| down down3 | **128（更少）** | 2 blocks | **29.43%** | **5.95 ms** |

**mma2 用了更多寄存器、驻留更少 block，却更快。** 所以「寄存器压制占用率」虽是真读数，
**却不是可动的杠杆**；可动的杠杆是**别拿寄存器当权重暂存缓冲**。
**暂存角色主导，占用率角色次要。** 这也解释了 E-SELF-02 为何整条路（提占用率）全负：
方向对的是「腾出寄存器」，但当时用 `.minnctapersm` 强压导致编译期 spill，等于把暂存挤到 local memory，更慢。

---

## 4. 混淆与边界（必须说清）

1. **mma2 同时改了两件事**：权重走 cp.async **且** `M2_COLS=256`（v1 是 64），
   程序数 43344 → 10836（少 4 倍），激活重复 staging 也随之少 4 倍（L2 请求 −42%）。
   **故 gate/up 的 −8.7% 不能全部归给 cp.async。**
2. **down3 是更干净的检验**：它的列块 20（v1 是 40，少 2 倍），但**总程序数 246400 vs 32800，多 7.5 倍**
   ——因为它**不接受 work list**，grid=(ceil(maxm/64)=32, N/256=20, nexp_max=385)，靠 `rows_sh[0]<0` 早退。
   **在程序数多 7.5 倍的劣势下仍快 8.8%，且首因 long_scoreboard 从 33.1% 腰斩到 18.5%**
   ——这正是 cp.async 盖内存延迟的靶向效果。**机制归因在 down3 上强得多。**
3. **barrier 在 down3 上升**（6.6% → 25.6%）：cp.async 环每级一个栅栏，这是新结构的**代价**，不是病。
   净账仍为正。
4. **仍未吃满任何一项**：改后 DRAM 61–65%、Tensor 46%、SM 48–49%。
   `math_pipe_throttle` 从 8–10% 升到 19–23%，说明**正在向算力约束靠近**，但还没到。
   **同一杠杆还有余量**（更多 stage / 更宽 tile / mma3）。
5. **全部离线单核**，未在生产多路（head 最小空闲 4.8 GiB）下复验。

---

## 5. 生产可用性：两行处境不同

| | 能否跑我方生产形状 | 障碍 |
|---|---|---|
| **mma2**（gate/up，25.79% 那行） | **不能** | 硬约束 `N % M2_COLS(256) == 0`；我方 TP2 本地 I=1152=256×4.5 **不满足**。本轮被迫两臂同跑全宽 N=2304 |
| **down3**（down，12.59% 那行） | **能** | N=D=5120=256×20 ✓、K=I=1152 %64 ✓。但输出 bf16 `Y[P,N]`、需换 `combine_y` 而非 `down_combine`，**非 drop-in**；且不吃 work list，程序数多 7.5 倍 |

**Amdahl 上界（以 prefill 中位 2084.42 tok/s、窗口 4.972654 s 为底，禁止比例连乘）：**
- down3 若落地：省 626.15 ms × 6.79% ≈ **42.5 ms** ⇒ 窗口 −0.855% ⇒ **约 +0.9%，2084 → 约 2102**
- mma2 若解决形状后同幅生效：省 1282.39 ms × 8.7% ≈ **111.6 ms** ⇒ **约 +2.3%，2084 → 约 2132**
- 两者都落地：**约 +3.2%，2084 → 约 2151**

**即：机制被证实，但现成可摘的果子很小。** 这与 E-SELF-04 复盘算出的 Amdahl 一致——
**单行归零也够不到 4000**。用户已明确「不要纠结 4000，先把某项硬件吃满」，
故本轮的价值是**确认了杠杆方向并证明它有效**，而不是拿到了大收益。

---

## 6. 下一步（按判决力排序）

1. **分离 gate/up 的混淆**：mma2 `rows=128`（`<CB,2,10,128>` 已实例化）把程序数再减半，
   同 cp.async 同列宽 ⇒ 若继续变快则摊销有效，若持平则 cp.async 主导。
2. **mma3**（`:1311`）：同样 `N%256`，但 `K%128`、支持 `g==4 && N%512` 的 wide 变体，
   shared 结构不同（`m3_rows_bytes` 双缓冲 + `m3_rotate`）。可作第三个点。
3. **把同一杠杆推到底**：stage 数 S、tile 宽度是 constexpr，需改源重编（78.5 s/次）。
   目标是把 DRAM 从 65% 推向 90%+ 或把 Tensor 推向饱和 —— **这才是用户要的「吃满」**。
4. **down3 落地的工程账**：换 `combine_y`、bf16 输出、无 work list 的程序数膨胀，
   需先过精度闸与生产多路复验，再谈 +0.9%。

---

## 7. 复现

```bash
# Spark head，停服后（avail 118 GiB）
docker run -d --name exl3-probe --gpus all --entrypoint bash \
  -v /home/ztxjzh/ptxwork:/home/ztxjzh/ptxwork -e HOME=/home/ztxjzh \
  --cap-add SYS_ADMIN tensorfold-dsv41:exl3-tp2-8348338a8d90 -c "sleep 7200"

docker exec exl3-probe bash -c "cd ~/ptxwork && python3 probe-mma2.py --n 2304 --reps 8 --out probe-mma2-ab.json"
docker exec exl3-probe bash -c "cd ~/ptxwork && python3 probe-down3.py --reps 8 --out probe-down3-ab.json"
docker exec exl3-probe bash -c "cd ~/ptxwork && bash probe-mma2-ncu.sh && bash probe-down3-ncu.sh"

python3 ~/ptxwork/probe-ab-parse.py --a-file m2-v1.csv  --b-file m2-mma2.csv  --a-label v1 --b-label mma2
python3 ~/ptxwork/probe-ab-parse.py --a-file d3-v1.csv  --b-file d3-down3.csv --a-label v1 --b-label down3
```

原始证据：`raw/`（4 个 ncu CSV、2 个判决文本、2 个 A/B JSON）。
工具：`tools/probe-mma2.py`、`probe-down3.py`、`probe-mma2-ncu.sh`、`probe-down3-ncu.sh`、`probe-ab-parse.py`。