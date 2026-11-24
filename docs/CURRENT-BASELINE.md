# 当前基线（唯一权威口径）

**本页是项目状态与数值的唯一口径来源。** 其他文档若与本页冲突，**以本页为准**。
方法论与纪律见 [开发手册](<DEVELOPMENT-HANDBOOK.md>)（必读）；逐实验证据见 `runs/`；历史过程见 [航海日志](<performance/02-航海日志.md>)。

**基线标识** `base-alias`　**最后核对** 2026-11-24（Asia/Shanghai）　**对应 commit** `0986e56`
**状态** 已生产验证、已入库、开机自启已核验。

---

## 1. 基线由什么构成

| 层 | 值 | 说明 |
|---|---|---|
| 引擎上游 pin | `fbbc97132cd004237d9887b8d3185896efaedc5c` | `https://github.com/bertholomus/TensorFold.git` |
| 我方 patch | **53 个**，`zig/patches/0001..0068` | 按序 `git apply --3way`，**53/53 干净落地零冲突** |
| kit | `fdde3285fd761add38e6d4f703f5402936f37220` | `deepseek-v4.1-tensorfold-tp2-2xgb10` |
| 生产镜像 | **`tensorfold-dsv41:base-alias`** | 两节点均有；基镜像 `tensorfold-dsv41:exl3-tp2-8348338a8d90` |
| 权重 | EXL3 **2.9 bpw** + EAQ v1.1 + ablit | 未改量化、未改数学内核 |
| 权威配置 | `scripts/local.sh.authoritative`（git 内） | 部署时物化为 `scripts/local.sh`（后者被 gitignore） |

**复现基线树**：pin 的 worktree + 53 patch 按序落地。校验用 `tools/verify-baseline.sh`。

> **两条路线的关系（明确声明）**：`v0.7.5 换底座`路线（E-PORT-01/3a/r2/3b/3c）**已搁置**。
> 同场实测证明它无回退但也只 +0.11%/+0.36%，而主线已改为「**我方基线 + 逐个吸收社区增量**」。
> **生产跑的是 `base-alias`（旧 pin 路线），不是 `v075-3b`。**
> `zig/patches-v075/`、`docs/DEPLOY-v075-NOTES.md`、`runs/forge-20261124-port-*` 均为**搁置路线的存档**，
> 保留备查，**不代表当前基线**。

---

## 2. 我方 53 个改动全清单（按功能分组）

每 patch 的开关、默认值与作用见 [`zig/patches/README.md`](<../zig/patches/README.md>)。

| 组 | 数量 | patch |
|---|---:|---|
| **KV 分片与注意力** | 6 | 0001 indexer-split、0015 kv-shard-attn、0018 kv-shard-lanes、0022 kv-shard-ring、0042 kv-shard-pieces、0038 aligned-index-views |
| **Spill 与缓存分层** | 5 | 0013 core-spill-store、0014 core-cache-tier、0016 spill-blocks、0019 spill-kv-shard、0021 spill-pictures |
| **内存与池** | 5 | 0008 pool-window、0011 cuda-budget-parked-pages、0023 server-prepare-memory、0040 restore-room、**0068 prompt-alias** |
| **量化与权重格式** | 3 | 0024 mixed-pack、0025 widen-bf16、0017 kit-row-split |
| **投机与草稿** | 2 | 0049 first-draft-depth、0050 draft-calibration |
| **prefill 调度** | 5 | 0005 prefill-interleave、0026 plain-rows、0051 prompt-rows、0041 service-between-pieces、0045 cut-in-tail |
| **CUDA graph / RDMA / 通信** | 5 | 0033 graph-cap、0010 rdma-idle-nap、0037 rendezvous-ports-early、0020 pair-broken、0059 lane-thread-binds-context |
| **Engram** | 1 | 0035 engram-lead-failure-paths |
| **API 与前端协议** | 14 | 0027 tool-call-prose、0028 plain-request-text、0029 anthropic-budget-default、0030 anthropic-context-management-lenient、0031 anthropic-system-anywhere、0032 stream-keepalive-ping、0046 server-get-delete-body-discard、0047 replayed-reasoning、0048 stream-tool-calls、0057 drop-reasoning-history、0058 responses-include-encrypted、0064 dsml-opener-short-payload、0065 dsml-value-holds-closer、0067 stream-single-call-heartbeat |
| **图像与多模态** | 3 | 0043 picture-resume-skip、0056 server-picture-prepare-scratch、0066 picture-resume-metrics |
| **遥测** | 1 | 0036 metrics-cached-prompt-tokens |
| **基础修复** | 3 | 0004 call-gate、0012 dsv41-base-fixes、0034 rebase-followups |

**其中兑现了可量化收益的关键改动**：

| 改动 | 收益（实测） | 出处 |
|---|---|---|
| `ENGRAM_AIO=0`（config.sh 默认，非 patch） | prefill **+19.5% @8K / +20.1% @41K** | `runs/forge-20261124-engram-aio` |
| **0068 `TF_DS_PROMPT_ALIAS=1`** | arena **3766 → 2208 MiB（−1558）**，avail 8→10 GiB；性能中性；**位一致 31/31 IDENTICAL** | `runs/forge-20261124-alias-audit` |
| KV 分片（0015/0018/0022/0042） | KV **532 MiB/rank** vs 复制 893 MiB，**省 40.4%** | `runs/forge-20261124-port-3b` |
| 0033 graph-cap | 修 CUDA graph 无上限增长（长跑内存） | `zig/patches/README.md` |
| `TF_DS_L2_PREFETCH=1`（基线自有） | **+6.24%/+6.40% 单流，+3.82%/+3.87% 四路** | `runs/forge-20261124-decode-traffic` §4 |

---

## 3. 生产运行态配置

**显式设置**（`scripts/local.sh.authoritative`）：

```
WORKER=ztxjzh@192.168.100.11      FABRIC_PEER=192.168.100.11
NFS_SERVER=192.168.100.10         NCCL_RAILS=2
RDMA=1  EAQ_V1_1=1  ABLIT=1       HF_HUB_OFFLINE=1
HOST=0.0.0.0  PORT=8888  API_KEY=1
CONTEXT=524288  SPILL_GIB=64  KV_SHARD=1
HEAD_ENGRAM_DIR=/home/ztxjzh/.cache/dsv41-tensorfold/engram-1bae0e10
PREFILL_INTERLEAVE_MS=0  (+ export TF_DS_PREFILL_INTERLEAVE_MS=0)
KIT_REF=fdde3285fd761add38e6d4f703f5402936f37220
ROUND_ROWS=16  PARALLEL=4
IMAGE=tensorfold-dsv41:base-alias
export TF_DS_PROMPT_ALIAS=1
```

**由 `scripts/config.sh` 默认提供（同样承重，不可随意改）**：

| 项 | 值 | 备注 |
|---|---:|---|
| `DRAFTS` | 1 | checkpoint 自带 DSpark 草稿块 |
| `GRAPHS` / `GRAPH_CAP` | 1 / 128 | 0033 |
| `SIDE` | 1 | mix 侧工作独立 stream |
| **`PREFETCH`** | **1** | **关掉亏 6.3% 单流，见 §2** |
| **`ENGRAM_AIO`** | **0** | **不是上游的 1，见 §2** |
| `REPLAY_FLOOR` | 1 | |
| `POOL` | 1052672 | 由 `scripts/port.sh` 从 CONTEXT 推导 |

**已判决不得设为默认的开关**（详见 §5 与手册死路清单）：
`TF_DS_SOLO_OVERHEAD`、`TF_DS_NO_COPIES`、`TF_DS_IDX_SPLIT`、L2 budget 参数改写。

---

## 4. 权威数值（同场实测，可直接相减）

> **纪律：任何 A/B 必须同场重测对照臂，禁止复用历史基线数值。**
> 本机同日漂移实测可达 **1%**（跨会话 N=4 观察到 0.44% 单调漂移）。

**当前地板 = `base-alias` + `PROMPT_ALIAS=1`**：

| 指标 | 值 |
|---|---:|
| **N=1 decode** | **59.265 tok/s** |
| **N=4 聚合** | **107.621 tok/s** |
| prefill 中位 | 2084.42 tok/s |
| arena | **2208 MiB** |
| 运行态 avail | **10 GiB** |
| 停服后 avail | 118 GiB |

同场对照表（2026-11-24，同一硬件同日）：

| 配置 | N=1 | N=4 聚合 | prefill 中位 | arena MiB |
|---|---:|---:|---:|---:|
| 旧引擎无分片 | 60.078 | 109.783 | 2087.47 | — |
| 旧引擎有分片（同场锚点） | 59.223 | 107.375 | — | 3766 |
| 3b 有分片（搁置路线） | 59.289 | 107.762 | 2061.84 | 3872 |
| base-alias OFF | 59.236 | 107.518 | 2083.16 | 3766 |
| **base-alias ON（当前地板）** | **59.265** | **107.621** | **2084.42** | **2208** |

**硬件屋顶（实测，E-ROOF-01）**：DRAM 可持续读 **246.2 GB/s**（官方 273，达成 90.2%）；
FP16 稠密 GEMM **100.82 TFLOPS**；机器平衡 409.4 FLOP/byte。

**历史值 `59.746 / 109.226 / 2085.24` 已作废为判决依据**，仅可作趋势参考。

**内存红线**：生产多路 head 最小空闲 **4.8 GiB**。
**本页所有数值均为空载单/四路测得，生产多路未复验。**

---

## 5. 判决台账（吸星大法）

流程见手册 §2。**先读码判是否有益 → 定吸收方式 → 才做 A/B → 确认收益才并入基线。**

| 编号 | 候选 | 判决 | 依据 |
|---|---|---|---|
| E-ABS-01 | `TF_DS_PROMPT_ALIAS`（上游 `3b902fd21`） | ✅ **吸收 → patch 0068** | arena −1558 MiB，位一致 31/31，性能中性符合「容量特性不提速」预期 |
| E-ABS-02 | `TF_DS_NO_COPIES`（基线自有） | ❌ **否决** | 常规探针 +0.3~0.6%，但**重复文本单流 −4.87%/−4.46%**（探针选择偏差） |
| — | `TF_DS_SOLO_OVERHEAD`（基线自有） | ❌ 否决 | N=1 **−2.22%**，草稿更深但接受率 0.5646→0.5417 |
| E-ABS-03 | `28dbfe128` L2 参数旋钮 | ❌ **否决** | 八臂扫描（加倍/减半/配速0）全噪声；上游 TP4 零结论在 TP2 复现 |
| E-ABS-03 | `bfa786f42` share_check | ⏸ 跳过 | 我方无 `TF_FILL_CHUNKS`，pass 本就整段跑 |
| B 步 | `TF_DS_PASS_FROM`（`37b7d2651`+`3befde133`） | ⏸ **降级** | <8192 构造上零收益；10752 探针 4096vs2048 仅 +0.05%；内存缺口 714 MiB |
| E-SELF-01 | 自研「给 router/topk 加 load staging」 | ❌ **否决** | 靶子只占整轮 2%（`m_rt` 629.9 µs vs `moe` 12909 µs）。⚠️ 原写「基线 `lane_gemv` 已是 cp.async 7 级流水」**读错家族**（那是 Nemotron，我方不调用），已由 E-SELF-02 更正 |
| — | `TF_DS_IDX_SPLIT` | ❌ 已否决 | −1.9% |
| — | `TF_FILL_MANY`（`452ee06dd`） | ⏸ 未审 | 基线确无；属调度准入非 decode 字节，优先级低 |
| — | `TF_DS_SERVED_K`（`799cc6031`/`0133884b7`） | ⏸ 未审 | 基线确无；影响 tok/轮，TP4 标注需查门控 |
| **E-SELF-02** | **提高专家内核占用率** | ❌ **路 A 实测否决**；路 B 未做 | 机制数字成立（prefill down REG:255、零 spill、静态 16.7%、寄存器绑定），**源码公开**（上游 `c99083245`）。路 A 两个低寄存器变体都把静态占用率抬到 25% 却都掉吞吐：**`g=1` −0.92%（4/4 配对全负）、`nt=4,pf=2` −2.38%（2/2 全负）**。见 §5.1。**路 B 已由 E-SELF-04 编译期否决** |
| **E-SELF-04** | **位一致闸 + 路 B + G×NT 不变线** | ✅ 闸通过；❌ 路 B 与 G=4 均否决；**根因已修正** | **位一致闸通过**：PTX 往返 86 内核资源账全等 + **229656 条 SASS 全等**；源码侧 56 s 重建 served 两件各 4096 条 IDENTICAL（判据是指令+资源一致，**整文件字节不可达亦不必要**）。**路 B 编译期否决**：`.minnctapersm 3/4` → REG 255→168/128 但 **STACK 0→432/656 B**。**G=4/NT=4 实测 −1.84%**（2/2 配对全负，零 spill、程序数不变）。**同场 2×2 因子实验定案**：解码量×2 **−0.81%**、激活量×2 **−2.25%** ⇒ **激活侧边际代价约 2.8 倍，且解码减半无收益** ⇒ **E-SELF-03 §7「造币工序是绑定约束」被否证**（事实描述仍成立）。**新工具链：改 PTX 20 s / 单编 .cu 66 s，换 cubin 不重建引擎**。见 §5.2。**⚠️ 2026-11-25 复盘：本行的「因子分解」与「激活侧是绑定约束」结论已作废**（三处前提算错：MG=64 是容量非活跃重复倍数、真实 ∈[1.16,2.13]；程序数并非恒定而是随 G 增长；测试格均同时改了程序数）。**且 down 仅占窗口 12.59%，gate/up 占 25.79%**，down 归零上界仅 +14.4%。**当下主因仍未定位**。详见 `runs/forge-20261124-self-rows-gn/REPORT.md` §8 |

**累计：吸收 1、否决 5、搁置/未审 4。吞吐增量为零，内存地板抬高 1558 MiB。**
这符合总纲：**没有可验证增量时，基线就是最优解，不为跟上而跟上。**

**decode 分段实测（E-SELF-01，profile 口径只看占比）**：`moe` 12909 µs/轮（stretch 口径 **78.2%**），
路由 `m_rt` 仅 629.9 µs（整轮约 2%）。**decode MoE 估算已达约 80% DRAM 屋顶**（估算，非实测流量）。

### 5.1 缺口性质已定：低占用延迟受限（**不是**未分离）

**E-ROOF-01 提出的 2.00× 缺口，两假设早已由 E-DRAM-01 r2 分离**
（`runs/forge-20261123-dram/REPORT.md`，零注入 PM 采样，引擎照常运行）：

- **H1（tile 重读打到 DRAM、已在屋顶）否证**：最大 chunk 941.6 ms 内 DRAM 均值 **121.5 GB/s = 屋顶 49.3%**，
  >200 GB/s 仅占时 9.8%。H1 要求的「42.8% 时间贴近 246 GB/s」形态不存在。
- **H2（内核效率缺口）成立**。重读是真的但**被 L2 吸收**（请求 742.2 GB/s，命中率 **84.5%**），未变成 DRAM 流量。
- **机制：低占用延迟受限**。三条饱和度同时低（DRAM 46.9%、SM Issue 36.2%、Tensor 37.0%），
  **Occupancy 中位仅 17%** ⇒ **在飞的内存请求不够多，盖不住延迟**。

**静态限制因子**（**E-SELF-02 已用 `cuobjdump -res-usage` 从二进制核实**）：

| 内核 | regs/thread | threads/block | shared | spill | 驻留上界 |
|---|---:|---:|---:|---:|---|
| **prefill down** `tf_exl3x::grouped_rows_kernel<2,8,4,1,2,16,2,false>` | **255** | 128 | 33920 B | **0** | **8/48 warps = 16.7%** |
| decode `tf_exl3x::grouped_cp_kernel<2,8,4,3,2,10,1/2>` | 128 | 128 | 17472 B | 0 | 16/48 = 33.3% |

寄存器绑定（shared 允许 3 blocks，寄存器只允许 2）⇒ **静态 16.7% 与 achieved 17% 一致，机制成立**。
（旧表里「gateup 106 regs / 256 threads / 44288 B」那行**家族串了**：那是引擎侧 **Nemotron** 的
`tf_experts_prefill::prefill_kernel`（实测 98 regs），我方 dsv41 不调用它。）

**⇒ 静态占用率不是充分的收益条件（E-SELF-02 路 A 已实测否决）。**
源码公开在上游 `c99083245` 的 `src/tensorfold/cuda/exl3/experts_grouped.cuh`（2283 行）
+ `experts_cb2.cu`，cubin 里也已有低寄存器变体，故路 A 无需重建 cubin 即可试。
实测两个变体**都把静态占用率 16.7%→25%，却都掉吞吐**：

| 变体 | REG | 静态占用率 | 同场 prefill 配对差 |
|---|---:|---|---|
| 基线 `nt=8,pf=1,g=2` | 255 | 16.7% | — |
| `g=1` | 168 | 25% | **−0.92%**（4/4 全负，复用减半⇒字节放大） |
| `nt=4,pf=2,g=2` | 168 | 25% | **−2.38%**（2/2 全负，字节不变⇒亏在工作粒度/ILP） |

**寄存器账数字正确 ≠ 可回收收益。** 路 B（同形状加 `__launch_bounds__` minBlocks）是唯一不受
形状混淆的干净检验，但有实据的 spill 风险（cubin 受约束变体 STACK 达 1096 B），期望值应下调。
源码改动已回滚，仍 53 patch。详见 `runs/forge-20261124-self-occupancy/REPORT.md` §0.2。

**⇒ 根因已由 E-SELF-03 定位到代码（不是占用率）：**
我方 prefill down `grouped_rows_kernel` 的供给侧是 **`__ldg` 阻塞式 global→寄存器、
缓冲深度 PF=1**，缓冲 `pf[PF][NT][LW]` 就在寄存器里，而 255 个寄存器中
**约 128 个被累加器 `acc[G][NT][2][4]`（2×8×8）占死** ⇒ 缓冲无法加深 ⇒ 在飞请求不足
⇒ Tensor Pipe 仅 37% active、DRAM 仅 49.3% 屋顶。
**决定性对照**：同文件我方 **decode** 走的 `grouped_cp_kernel` 用
**cp.async → shared 多级环形 + `__launch_bounds__(W*32,4)`**，实测 **DRAM 82.8% 屋顶**。
同一硬件同一内存系统 ⇒ **「显存速度不够」被排除，差别在供给侧设计**。
**这也解释了路 A 为何必然失败：降寄存器与降复用在这个内核里是同一件事（寄存器被累加器占着）。**
杠杆：**shared 每 block 尚空 17280 B，可放 3–5 级 cp.async 缓冲且不花寄存器**，
与路 A 的失败机制正交；但 cubin 无 PF=2&NT=8&G=2 件，须自建。
详见 `runs/forge-20261124-self-backpressure/REPORT.md`。

> **更正批注（E-SELF-04，2026-11-25）：上面这条根因链的「绑定约束」判定已被实测否证。**
> 原文保留。E-SELF-03 §7 把「造币工序（解码 ALU/洗牌）」定为卡点，
> 但同场 2×2 因子实验显示：**解码量翻倍只 −0.81%，激活量翻倍 −2.25%**，
> 且**把解码量减半（G=4/NT=4，零 spill、程序数不变）换不到吞吐（−1.84%）**。
> ⇒ 解码 ALU **不是**绑定约束；**激活 X 的重复读取才是**：
> X 读取总量 = 列块数 × maxm × K，当前列块数 = 5120/128 = **40**，每份激活被读 40 遍；
> 权重字节反而有 L2 84.5% 命中兜底。
> **方向随之修正**：结构改造的目标不是「缓存解码后的成品」，而是
> **「缓存激活、让一个程序覆盖更多列」**，把列块数降下来。
> 已量化障碍：32 行 × 全 K 激活 = 73728 B > 2 blocks 的 51200 B 预算；
> NT=16 时 `red` = 65536 B 同样超预算；1 block/SM 放得下但占用率 8.3%
> （参照 decode 在 REG:128 下达 82.8% DRAM 屋顶，低占用不必然低吞吐）。
> 详见 `runs/forge-20261124-self-rows-gn/REPORT.md` §5.2–§5.4。
>
> **⚠️ 上面这条更正本身也已在 2026-11-25 复盘中被推翻，勿采用。** 它依据的「因子实验」并非单因子
> （三个测试格都同时改了程序数 ×1.47~×2.00），且「解码冗余 64 倍」是**容量口径**误读——
> 由 trace 的 grid 反解 p=14336、maxu=385 ⇒ 每专家均 37.24 行，
> **真实重复倍数 ∈ [1.16, 2.13]**，本来就没多少可省。
> 「沿 G·NT=16 程序数恒定」漏了工作列表的每专家 padding 地板项 `maxu`，实际随 G 线性增长。
> **另：down 只占窗口 12.59%（626.15 ms），gate/up 占 25.79%（1282.39 ms）**，
> down 归零 Amdahl 上界仅 **+14.4% → 2384 tok/s**，两者同时归零也只 3382。
> **结论：当下主因仍未定位；本轮是在 12.59% 的行上做局部灵敏度探测，
> 并把灵敏度错当成了占比与绑定约束。** 权威更正见该报告 §8。

### 5.2 编译期预筛闸与 cubin 直换（E-SELF-04 新工具链）

**位一致闸已通过两条独立路径**，据此确立「构建路径忠实」，之后的预筛结果才可信：

| 路径 | 耗时 | 判据 |
|---|---:|---|
| kit cubin 内嵌 PTX（`.nv_debug_ptx_txt`）还原后 `ptxas` 重汇编 | **20 s** | 86 内核资源账全等 + **229656 条 SASS 全等** |
| 公开源码单编 `experts_cb2.cu`（`nvcc -cubin`，无需 pybind） | **56 s** | 83 内核资源账全等、0 不同；served 两件各 **4096 条 IDENTICAL** |

**判据是指令 + 资源一致，不是整文件字节一致**——`ptxas` 复现不了 kit 的 `.debug_*` 段
（49554312 B vs 11695008 B），整文件 sha256 仅作记录。

编译坐标（读自源码）：`experts.py:129` `-O3 -lineinfo`；`build.py:33` `-gencode=arch=compute_121,code=sm_121`；
torch `COMMON_NVCC_FLAGS` 四个 `-D__CUDA_NO_*` + `--expt-relaxed-constexpr`；`-std=c++17`。
**`ops_build.py` 的 `--fmad=false --ftz=false` 只属算子 oracle，exl3 不用，混用会破坏一致性。**

**换 cubin 不重建引擎**：`TF_DS_KIT` 指 kit 目录；`start.sh` 不校验 kit 哈希（只 `prepare.sh:231` 校验）；
`worker_files_check` 只查存在/可读/非空；`FingerprintMismatch` 是跨 rank 一致性、不作废权重 cache；
worker 的 hf 卷是 head 的 NFS 只读共享 ⇒ **cubin 只需在 head 换一次**。
把基线与新变体编进**同一个超集 cubin**（基线内核验证仍逐条一致），A/B 就只是翻 env + 重启。

**真实形状（读自 shape-audit，非推算）**：down N=d=**5120**、K=i=**1152**、KT=72、每 warp **18 个 k-tile**。
**18 不是 MMA_KB=4 的倍数**，故 `exl3_experts.zig:207` 放行——**down 走 rows 是形状约束，不是选择**。

**实验后已还原**：`local.sh` 与 kit `experts_cb.cubin` 均回原件（sha `cdb154ba…`），生产回 `base-alias`。
`base-gn` 镜像与超集 cubin 备份留在两节点备下次实验。

**时钟杠杆永久关闭**：用户已主动降频，**升频实测无收益**；且官方 boost 是 **2.4 GHz**
（GB10 白皮书 Table 4），非 nvidia-smi 报的 3003 MHz，此前「只跑到 74.5%」是假缺口。
官方 **L2 = 24 MiB**（容量已知，带宽官方不给）。

**两条口径边界，不可越界**：
1. 上述占用率与饱和度测的是 **prefill**。**decode 占用率未测**，不得直接搬用。
2. 该 2× 缺口属 **prefill**，**不可当 decode 的可回收上界**。

**仍开放的是具体 stall 根因**（long scoreboard / barrier / 解码依赖链），
需 stall-reason 计数器，PM 采样给不了；**驻留态禁止 kernel replay**（已证爆内存）。

---

## 6. 开机自启链路（已核验）

| 环节 | 状态 | 核验命令 |
|---|---|---|
| head `Linger` | **yes** | `loginctl show-user ztxjzh --property=Linger` |
| user service | **enabled / WantedBy=default.target** | `systemctl --user is-enabled dsv41-tensorfold.service` |
| `scripts/local.sh` | 与权威副本一致 | `diff scripts/local.sh scripts/local.sh.authoritative` |
| 两节点 docker | **enabled** | `systemctl is-enabled docker` |
| 两节点镜像 | `base-alias` 均在 | `docker images \| grep base-alias` |
| head→worker 免密 | **KEY_OK** | `ssh -o BatchMode=yes ztxjzh@192.168.100.11 echo KEY_OK` |
| worker `Linger` | no（**不需要**） | worker 无 user service，由 head `start.sh` 远程 `docker run -d` 带起 |

**重启后会发生什么**：head 开机 → linger 拉起 user manager → `default.target` 拉起
`dsv41-tensorfold.service` → `start.sh` 读 `scripts/local.sh` → 起 head 容器并 ssh 起 worker 容器 → 约 50 s 就绪。

**单点风险与对策**：`scripts/local.sh` **被 gitignore**（含机器特定路径）。
对策：权威副本 `scripts/local.sh.authoritative` **在 git 内**，
`tools/deploy-baseline.sh` 负责物化并校验一致性。**若 local.sh 丢失或与权威副本不一致，部署脚本会拒绝启动并提示。**

**重启后自检**：`tools/check-boot.sh`（健康、镜像、alias 在位、avail、无残留实验开关）。

---

## 7. 复现与部署

```bash
# 1) 校验 53 patch 能在干净 pin 上按序落地
tools/verify-baseline.sh

# 2) 部署到两节点并启动（含配置物化、镜像核验、unit 安装、健康检查）
tools/deploy-baseline.sh

# 3) 重启后自检
tools/check-boot.sh
```

**从零重建引擎**（约 6–10 分钟，须先停服并断言 avail > 100 GiB）：见手册 §9 与
`docs/DEVELOPMENT-HANDBOOK.md` 的构建段；构建脚本模式与派生镜像 Dockerfile 见 `docs/DEPLOY-v075-NOTES.md`（**该文档属搁置路线，仅构建手法可复用**）。

**内核占用率诊断**（只读，不构建不启动）：`tools/kernel-resusage.sh`
—— kit cubin 用 `cuobjdump -res-usage`、引擎 kernel 用 `nvcc -Xptxas -v`。
**任何关于寄存器/占用率的论断都必须从这里取数，不得沿用文档数字**（手册 §7.4）。

---

## 8. 文档地图（口径优先级）

1. **本页** — 状态、配置、数值、判决的唯一口径
2. [开发手册](<DEVELOPMENT-HANDBOOK.md>) — 方法论与纪律的唯一口径（**必读**）
3. `runs/*/REPORT.md` — 逐实验原始证据与更正批注（**只增不改**）
4. [发现与结论](<performance/01-发现与结论.md>) / [航海日志](<performance/02-航海日志.md>) / [实验索引](<performance/03-实验与证据索引.md>) — 历史与索引
5. [BASELINE.md](<performance/BASELINE.md>) — `perf-baseline-20261124` 历史快照（**已被本页取代为当前口径**）
6. `docs/DEPLOY-v075-NOTES.md`、`zig/patches-v075/` — **搁置路线存档**