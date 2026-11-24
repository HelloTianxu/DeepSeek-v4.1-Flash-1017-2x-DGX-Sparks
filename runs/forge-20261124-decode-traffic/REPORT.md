# E-ABS-03 · 专项筛查：2211 个社区提交中「减少 decode 每轮内存搬运」的内容

**动机**：decode 是 DRAM 带宽瓶颈（本机屋顶实测 **246.2 GB/s**）。
单流吞吐的着力点只有两个：**每轮少搬字节**，或**每次搬运多产 token**。
本轮按这两条筛 `fbbc9713..e100ee8b`（v0.7.5）全部 2211 个提交。

**总结论**：**社区没有我方尚未吸收的、适用于 TP2 / ≤4 路的 decode 搬运优化。**
绝大多数该类改动**基线早已有**（再次印证「提交数是假象」）。
真正可做的是：①确认我方自有开关（已做，见 §4）；②吸收一把上游的**测量尺子**并在我方 TP2 上重扫（§5）；
③把两条**异后端思路**自研到 dsv41 CUDA 上（§6）。

---

## 1. 筛选漏斗

| 层 | 数量 | 说明 |
|---|---:|---|
| `pin..v0.7.5` 全部提交 | 2211 | 合并拓扑造成，绝大多数与我方无关 |
| 触及 `core/lanes/` + `core/round.zig` | 85 | 通用轮机制 |
| 触及 dsv41 `round/ops/lanes/windows.zig` | ~40 | **我方真实 decode 路径** |
| 触及 `kernels/cuda/exl3.cu` 等 | 15 | 我方量化 kernel |
| 关键词命中（bandwidth/fuse/once/prefetch/L2/traffic…） | ~60 | 含大量 Metal / 他家族 |
| **逐条读判决后剩余真候选** | **1** | `28dbfe128`（且为测量脚手架） |

---

## 2. 判决表：看似相关、实则不适用或已在基线

| 提交 | 标题要点 | 判决 | 依据 |
|---|---|---|---|
| `dd7c516a4` / `9630b3d81` | **window twin kernels：每个权重一个窗口只读一次** | ❌ 后端不符 | 改的是 `kernels/metal/core/tile_q*.metal` + `families/lfm2`。**纯 Metal / lfm2**，我方 CUDA dsv41 不走。**思路可借鉴 → §6** |
| `db7364e63` / `4f09b51fc` | **decode 尺寸 router/topk/add_moe_norm 让 load 保持在飞行中**（the same bits） | ❌ 家族不符 | 改 `nemotron_norms.cu` / `nemotron_route.cu` / nemotron glue。**思路可借鉴 → §6** |
| `dc9d8bb6c` / `8fe038e37` | decode 尺寸 gated norm 用 32 线程块（`launch.rowThreads`） | ❌ 我方不调用 | 改共享 `src/cuda/{launch,norms}.zig`，但 **dsv41 无该 import，基线 `launch.zig` 无 `rowThreads`** |
| `51426b8e6` | EXLR8 trellis decode 一词两权重、tile 提前一格 | ❌ 量化不符 | 我方 exl3 **2.9bpw**，非 EXLR8 |
| `06afb468e` / `c79c714ca` / `d27e3d422` | 64 行轮 / 32 流池 | ❌ 规模不符 | 用户 **≤4 路**；E-KIT-01 已证 >4 路才适用 |
| `c7aef7339` / `ffb3f1e99` | parity 专家切分、**decode gate/up 合成一次 launch** | ❌ 拓扑不符 | `TF_DS_2D_GU=parity` 明写 **four nodes**；我方 TP2 |
| `09bc7166f` | 2D MoE 保留下层 L2 prefetch | ❌ 拓扑不符 | TP4 2D MoE 路径 |
| `413986cde` | **paced L2 prefetch 原始实现**（3us 延迟、8KiB 块、230 GB/s） | ✅ **基线已有** | `TF_DS_L2_PREFETCH` + `tf_ds_l2_prefetch_kernel` 在位，且 `PREFETCH` 默认 **1** |
| `2130c4fa6` | mix 侧工作独立 stream + paced prefetch | ✅ 基线已有 | `SIDE=1` / `PREFETCH=1` |
| `f660ca5f7` | **轮尾：drafter pass graphs、word copies、device argmax** | ✅ 基线已有 | `draft.PassGraphs`（draft.zig:113）、`model.pgraphs`、argmax 见 6 文件 |
| `d429bf906` | indexer 全设备端选择（消除每轮 host 往返） | ✅ 基线已有 | 同批 perf 已入 pin 的另一分支线 |
| `d0c64a077` | `TF_DS_IDX_SPLIT` | ✅ 基线已有且**已否决** | 实测 −1.9% |
| `f493858a2` | `TF_DS_NO_COPIES` | ✅ 基线已有且**已否决** | E-ABS-02：重复文本单流 −4.7% |
| `fd9bbf5b9` / `3c6b217f0` | `TF_DS_SOLO_OVERHEAD` | ✅ 基线已有且**已否决** | 实测 −2.22% |
| `bfa786f42` | `share_check` | ⏸ 条件候选 | 我方无 `TF_FILL_CHUNKS`，pass 本就整段跑 |
| `452ee06dd` | `TF_FILL_MANY`（fill 期间准入请求） | ⏸ 非搬运 | 基线确无（grep=0），但属**调度/准入**，非 decode 字节；默认 off |
| `37b7d2651` / `3befde133` | `TF_DS_PASS_FROM` | ⏸ 已降级 | prefill 特性；<8192 构造上零收益，内存缺口 714 MiB |
| `a43306474` | `TF_DS_WINDOW` | ✅ 基线已有 | grep 命中 4 文件 |
| `799cc6031` / `0133884b7` | `TF_DS_SERVED_K`（单流时 served lane 草稿选择） | ⏸ 待审 | 基线确无（grep=0）。影响 **tok/轮** 而非字节；TP4 标注，需查门控 |

---

## 3. 方法论收获（已补入手册）

**「社区有 2211 个新提交」几乎不含信息量。** 判断某特性是否为增量，
唯一可靠做法是**在基线树按 env / 符号名 grep**，而不是数 `git log pin..upstream`。
本轮 ~40 条 dsv41 decode 路径提交里，**prefetch、side stream、drafter graphs、device argmax、
indexer 设备选择、idx_split、no_copies、solo_overhead 全部基线已有**。

**筛选必须先过「后端 / 家族 / 拓扑 / 量化 / 规模」五道闸**：
Metal ≠ CUDA、Nemotron ≠ dsv41、TP4 ≠ TP2、EXLR8 ≠ exl3-2.9bpw、>4 路 ≠ 我方负载。
本轮 ~60 条关键词命中里，**过半死在这五道闸上**。

---

## 4. 顺手确认我方自有开关：`TF_DS_L2_PREFETCH`（从未 A/B 过）

发现 `config.sh:168` `PREFETCH="${PREFETCH:-1}"` —— **一直开着，但和 SOLO_OVERHEAD / NO_COPIES 一样从未验证**。
传参链路：`start.sh:339` `TF_DS_L2_PREFETCH=$PREFETCH`；
**注意 `start.sh:291` 把该变量排除在直通之外，必须改 `PREFETCH`，直接 export 无效。**

交替 A/B/A/B，同场、同一 `base-alias` 二进制，仅 env 不同：

| | A1 (ON) | B1 (OFF) | A2 (ON) | B2 (OFF) | 配对差 |
|---|---:|---:|---:|---:|---:|
| **N=1** | 59.129 | 55.655 | 59.095 | 55.538 | **+6.24% / +6.40%** |
| N=1 tok/轮 | 2.0157 | 2.0157 | 2.0157 | 2.0157 | 恒等 |
| **N=4 聚合** | 107.433 | 103.481 | 107.481 | 103.480 | **+3.82% / +3.87%** |
| N=4 轮/秒 | 56.467 | 53.693 | 56.202 | 53.738 | — |

- 同配置内散布：N=1 **0.06% / 0.21%**，N=4 **0.04% / 0.001%**。效应约为散布的 **30 倍**。
- **N=1 tok/轮 四次恒等 → 纯轮/秒效应，纯内存路径**，与投机机制无关。

**两个结论**：
1. **我方这个默认是对的**，值 +6.3% 单流 / +3.8% 四路。这是**确认**，不是新增收益，基线不动。
2. **decode 内存路径确实有活头**：单个 prefetch 开关就能摆动 6.3%。
   这为 §6 的自研方向提供了实据——不是空谈。

---

## 5. 唯一吸收的候选：`28dbfe128`（测量脚手架，但上游零结论不可搬）

**它做什么**：把 5 个 prefetch 硬编码常量变成 env 可调，并加 skip 位与 clock ratio：
`TF_DS_L2_ATTN_MB` / `L2_MOE_MB` / `L2_NEXT_MB` / `L2_STRETCH_END_MB`（默认 16/8/16/4 MiB）、
`TF_EXL3_L2_DELAY_NS`（3000）、`TF_EXL3_L2_RATE_GBPS`（230）、`TF_DS_L2_SKIP`（位1 attn / 位2 moe / 位4 next）、`TF_DS_CLOCK_RATIO`。

**为什么本身不提速**：commit body 明写
> "Defaults unchanged: the same launches. **Slot Zt6: the settings swept on real one-stream decoding move nothing past noise.**"

即上游自己扫过，**噪声内**。默认行为逐 launch 相同 → **构造上零增量**。

**为什么仍值得吸收**：上游那套 budget 与配速是在 **TP4** 上调的，
**每 rank 权重字节数只有我方 TP2 的一半**；且其配速常量 230 GB/s 与我方实测屋顶 246.2 GB/s 不同。
**差异恰好落在决定性的那个维度（每 rank 搬运量）上，故其零结论不能外推到我方。**
§4 已证我方 prefetch 敏感度高达 6.3%，说明这把尺子在我方是活的。

**吸收方式**：cherry-pick（39 行，改动小且默认中性，风险最低）。
- 2 处冲突均为**纯叠加**，两边都保留：
  ①我方 `prompt_rows.validWindow/validGemm` + `window_rows/gemm_rows` **与**上游 L2 env 读取；
  ②我方 `clk_shard`（KV 分片遥测）**与**上游可调 `clk_ratio`。
- **重点核查**：上游删除了 `const prefetch_budget`，须确认我方 52 patch 未新增引用它的调用点 ——
  实测 `grep prefetch_budget zig/src/` **为空**，5 个 `rd.prefetch` 调用点全部转换，无悬空引用。
- 符号核对：`round`(model.zig:16) / `ops_mod`(:24) / `std.c.getenv` 均已在用。
- 基线树 commit **`79dfbe7d0`**。

**验收计划**：默认臂必须复现 §4 的 ON 数值（同场重测，不复用历史），
确认中性后再逐项扫 budget / 配速 / skip 位。**任何一项要留下，须过同场交替 A/B。**

### 5.1 实测结果：**否决**（上游零结论在我方 TP2 复现）

cherry-pick 基线树 `79dfbe7d0` → Spark 构建 **BUILD_OK** → 两节点镜像 `tensorfold-dsv41:base-l2`。

**先修一个真实集成缺口**：`TF_EXL3_L2_DELAY_NS` / `TF_EXL3_L2_RATE_GBPS` 前缀是 `TF_EXL3_`，
**不在 `start.sh:291` 直通白名单 `(TENSORFOLD|TF_DS|TF_SPILL|TF_FILL)_` 内，容器里一个都收不到**。
临时扩白名单后才测得到（C1 臂容器内确认拿到 `TF_EXL3_L2_RATE_GBPS=0`）。
**教训：吸收带 env 旋钮的特性，必须核对每个旋钮的前缀是否被我方启动脚本放行，否则测的是空开关。**

八臂实测（同一 `base-l2` 二进制，仅 env 不同；A 为默认对照）：

| 臂 | 设置 | N=1 | N=4 |
|---|---|---:|---:|
| A1 | 默认 16/8/16/4 MiB, 230 GB/s | 59.102 | 107.419 |
| B1 | **加倍** 32/16/32/8 | 59.016 | 106.869 |
| A2 | 默认 | 58.983 | 107.263 |
| C1 | **配速 0**（不节流，一次全预取） | 59.099 | 107.487 |
| A3 | 默认 | 59.077 | 107.070 |
| D1 | **减半** 8/4/8/2 | 59.087 | 106.965 |
| A4 | 默认 | 59.073 | 106.944 |
| D2 | 减半 8/4/8/2 | 59.055 | 107.096 |

**中性核对通过**：默认臂 N=1 59.102/58.983/59.077/59.073 与 §4 旧镜像同日的 59.129/59.095
相差 ≤0.2%，证明 cherry-pick 在默认下行为未变（与 commit body "the same launches" 一致）。

判决：
- **加倍**：N=1 跨零（−0.15% / +0.06%）= 噪声；N=4 两对同为负 **−0.51% / −0.37%**，约散布 3 倍 → 轻微有害。
- **减半**：N=1 四对全在 **±0.04%** 内；N=4 跨零（−0.10%/+0.02%、+0.02%/+0.14%）→ 噪声。
- **配速 0**：N=1 −0.005%/+0.20% 跨零；N=4 +0.06%/+0.21%，与散布同量级 → 噪声。
- N=1 tok/轮 **八臂恒为 2.0157**，确认这些旋钮只动内存路径、不动投机。

**结论：TP2 假设被证伪。** 「上游在 TP4 上调的、每 rank 字节数是我方一半」这个理由虽然成立，
但**实测表明该 budget 在 TP2 上同样不敏感**。上游的零结论在我方复现。

**故否决，不并入基线**：无可验证增量 → 不抬地板、不增 patch 面（维持 53 个）。
`start.sh` 白名单改动已回滚（避免留下指向不存在的 0069 的悬空注释）；
镜像 `base-l2` 与 `scripts/local.sh.l2-base` 留作后续诊断用，生产已切回 `base-alias`。

**会话内漂移实证**：四个默认臂 N=4 从 107.419 单调漂到 106.944（**0.44%**），
而待测效应量级仅 0.1–0.5%。**若拿首臂当对照、末臂当处理，会凭空得出「减半 −0.3%」的错误结论。**
这是 §3.1「必须同场重测对照臂」+ §3.5「小效应交替多轮」的又一次实证。

---

## 6. 可借鉴自研的两条思路（异后端，思路对路）

我方 CUDA dsv41 用不上其代码，但机制正对 DRAM 瓶颈：

1. **「每个权重一个窗口只读一次」**（`dd7c516a4`，Metal twin kernels）
   —— 把一个 decode 窗口内多行对同一权重的重复读取合并为一次。
   我方 `ROUND_ROWS=16`、≤4 路，窗口内行数少，收益上限需先算；
   但这是**减少字节**的正统方向，且 §4 证明该路径敏感。

2. **「让 load 保持在飞行中」**（`db7364e63`，Nemotron router/topk/add_moe_norm，位精确）
   —— 提高内存级并行度（MLP），把延迟受限的小 kernel 变成带宽受限。
   我方 dsv41 有自己的 router / topk / norm kernel，**同类算子、同类瓶颈、同一块 GPU**。
   位精确改法（不动算术、只动加载调度）风险低，适合先做。

**这两条属「启发我做一个更好的实现」，不是 cherry-pick。需先读我方对应 kernel 再定方案。**

---

## 7. 待办

- [ ] `28dbfe128` 构建 → 默认臂中性核对 → TP2 参数扫描 → 判决
- [ ] `TF_DS_SERVED_K` 读码判门控（基线确无，影响 tok/轮）
- [ ] §6 两条自研方向：先读 dsv41 自有 router/topk/norm kernel 与窗口权重读取路径
- [ ] `TF_FILL_MANY` 读码（调度类，非搬运，优先级低）