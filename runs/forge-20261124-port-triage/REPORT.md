# E-PORT-01 · 步骤2：52 个 patch 三分类，定出步骤3 的承重子集

**日期**：2026-11-24　**性质**：只读分析 + 一次配置级实测（已退回）
**生产状态**：全程未改引擎；结束时 `KIT_REF=fdde3285 / PARALLEL=4 / ROUND_ROWS=16 / IDX_SPLIT unset`，arena 3766 MiB，服务 active

---

## 0. 结论先行

**① 用户对步骤3 的重新定义是对的，且已确证。** 裸 v0.7.5（`e100ee8b`）**加载不了我方 EAQ v1.1 checkpoint**：

| 关键字 | v0.7.5 命中 |
|---|---|
| `mixed pack` | **0** |
| `plain_entries` | **0** |
| `plain_rows` | **0** |
| `linearBf16` | 仅 `cublas.zig` / `vit.zig`（vision tower），**不在权重加载路径** |

我方 checkpoint 是 EAQ v1.1 混合包：**27 个 BF16 indexer/compressor linear**（8 `attn.wq_a`、8 `attn.indexer.wq_b`、4 `attn.compressor.wkv`、3 `attn.compressor.wgate`、4 `attn.indexer.wk`）必须走 cuBLAS plain BF16，而非 EXL3 trellis。上游 v0.7.x 服务的是纯 EXL3 2.5bpw，没有这条路径。

⇒ **步骤3 必须 = v0.7.5 引擎 + 承重 patch 子集**，不存在"干净上游基线"这个选项。

**② patch 是依赖链，不能任意挑。** README 原文：

> "Remove them from the last one back, never one in the middle (each builds on those before it)."
> "KV sharding builds on `0008`'s pool: one `POOL`, cut into one part a Spark when `KV_SHARD=1`."

**③ 一处前轮错误更正**：我上一轮说「`TF_DS_IDX_SPLIT` 我方 pin 不支持，需换引擎」——**错**。pin 确实没有，但**我方 patch 0001 就是加它的**。当时只查 pin 源码、漏了自己的 patch。已实测（见 §3）。

---

## 1. 三分类表

### A 类 · 承重-格式（**必须重端口，否则 checkpoint 加载即错**）

| Patch | 作用 | 为何承重 |
|---|---|---|
| `0024-mixed-pack` | EAQ v1.1 的 27 个 BF16 linear 走 `cublas.Blas.linearBf16` | **上游 0 命中**；没有它这些层被当 EXL3 trellis |
| `0025-widen-bf16` | BF16 源精确展宽为 f32（高位放 BF16 bits，低位补零） | EAQ v1.1 的 8 个 `attn.indexer.weights_proj.weight` 是 BF16 而 plan 要 f32；缺它 rank-cache writer 报 `UnsupportedConversion` |
| `0026-plain-rows` | mixed pack 的 cuBLAS linear 固定 ≥N 行（默认16） | cuBLAS 按 GEMM 形状选 kernel 与 split-K，行数变化会改变 token bits。缺它 8 prompt 只匹配 4 |

**这 3 个是步骤3 的最小必要集，且必须按序。** 配套还有 `zig/patches-inputs/0001-mixed-pack-rank-cache.patch` 与 `0002-widen-bf16.patch`（第二个 checkout，建 rank-cache 用）。

### B 类 · 承重-配置与内存优势（**丢了净亏**）

| Patch | 作用 | 我方设置 |
|---|---|---|
| `0008-pool-window` | POOL 可大于 WINDOW | POOL=1052672 / CONTEXT=524288 |
| `0015` `0018` `0019` `0022` `0042` | KV 分片五件套（attn / lanes / spill配套 / ring交换 / pieces） | `KV_SHARD=1`，**省 40.4% KV**，上游全无 |
| `0013` `0014` `0016` `0021` | spill v2（块存储 / cache tier / 双rank协议 / 图片） | `SPILL_GIB=64` |
| `0040` `0041` `0045` | restore 等位 / pieces 间服务 / cut-in 按实算 token | spill+分片的配套 |

### C 类 · 承重-内存安全与稳定性（**与用户"不能爆内存"直接相关，优先级最高**）

| Patch | 作用 | 风险若不端口 |
|---|---|---|
| **`0011-cuda-budget-parked-pages`** | 启动预算计入内核 per-CPU parked pages | **镜像加载后首启会少算数 GiB** |
| **`0023-server-prepare-memory`** | 长 prompt 的 render/tokenize scratch 归还 + prepare 内存地板（3 GiB） | 8 路 ~1M token 恢复会把 rank0 MemAvailable 打到 0 |
| **`0033-graph-cap`** | CUDA graph cache 上限 128，LRU 淘汰 | **cache 连涨数天，设备内存 + ~1.6× 主机内存，直到 memguard 杀 rank0** |
| `0020-dsv41-pair-broken` | 坏对退出 75 而非挂死，配 systemd 重启 | 挂死等人工 |
| `0035-engram-lead-failure-paths` | Engram 读失败转 EngramRankFault 结束对，而非另一 rank 干等 | 同上 |
| `0056-server-picture-prepare-scratch` | 每张图在临时 arena 处理完即清 | 每 1024×650 截图占 ~16 MiB 直到请求结束 |
| `0012` `0034` `0037` `0038` `0049` `0059` | base fixes / rebase 收尾 / 早开端口 / 16字节对齐 / 首轮draft深度 / lane线程绑CUDA上下文 | 各类启动与正确性故障 |

### D 类 · API 与前端（17 个，功能需要，**不影响性能，可最后批量处理**）

`0004` `0027` `0028` `0029` `0030` `0031` `0032` `0036` `0046` `0047` `0048` `0057` `0058` `0064` `0065` `0066` `0067`

多为 Anthropic/Responses 前端兼容、DSML 工具调用流式、reasoning 字段回放。**与本轮性能目标无关**，但我方 Agent 业务依赖其中若干（工具调用、reasoning）。

### E 类 · 可丢或已失效（**候选精简**）

| Patch | 判定 | 依据 |
|---|---|---|
| `0005-prefill-interleave` | **实际已 off** | README：rebase 后 `TF_FILL_CHUNKS` 一设就自动 off（上游 pieces 做同样的事）。我方 `TF_FILL_CHUNKS=8` ⇒ 已 off。**但其 `Stream.between` hook 被 0014 使用，不能整删** |
| `0010-rdma-idle-nap` | **上游可能已有** | 它是 TensorFold PR #27 逻辑的 Zig 移植，需核实 e100ee8b 是否已含 |
| `0017-dsv41-kit-row-split` | **部分失效** | 它是「pinned kit 无法整机发射 MoE gate 时拆成 16 行」的 workaround。**新 kit fdde3285 原生带 RB 17..64**，该 workaround 对 decode 不再必要（但其 kit variant 工具与 spill plane staging kernel 仍在用） |
| `0050-draft-calibration` | 未启用 | 默认 off，我方未开 |
| `0051-prompt-rows` | 未触发 | E-REVIEW-01 已证对 2048 行主体不触发 |
| `0001-c7-indexer-split` | **保留默认，勿改** | 见 §3 实测 |

---

## 2. 步骤3 的重新定义（据本分类）

**目标**：v0.7.5 引擎（`e100ee8b`）+ 承重子集，与我方基线对比。

**最小可运行集（按依赖序）**：
```
0008 (pool-window，KV分片与spill的前提)
0011 (parked pages，内存安全)
0012 (base fixes)
0024 → 0025 → 0026 (EAQ v1.1 三件套，顺序不可换)
```
**⇒ 6 个 patch 即可让 v0.7.5 加载我方 checkpoint 并跑起来。** 这是"能不能跑"的最小集。

**完整对比集**（要与我方基线公平比较，还需 B+C 类）：再加 `0013/0014/0015/0016/0018/0019/0020/0021/0022/0023/0033/0040/0041/0042/0045` 等。

**建议分两阶段**：
1. **3a 最小集跑通**（6 patch）——只验"能加载、能出正确 token"，不开 KV_SHARD / SPILL，内存反而更省（KV 走 replicated 893 MiB，仅 +0.35 GiB）
2. **3b 补 B+C 类**——恢复 KV 分片与 spill，再与基线同口径比性能

**3a 的内存优势**：不开分片时 KV 893 MiB vs 我方 532 MiB，只多 0.35 GiB；而 arena 仍 3766 MiB。**比当前生产还宽松**，符合"避免引入内存问题"。

---

## 3. 附带实测：`TF_DS_IDX_SPLIT=1` 否决（E-IDXSPLIT-01）

更正前轮错误后顺手实测。patch 0001 的语义：

```zig
prompt.idx_split          = if (env) |v| eql(v,"1") else true;
prompt.idx_split_min_comp = if (env != null) 0 else 65536;
```
⇒ unset = 阈值 65536 压缩键；`1` = 阈值 0（全部合规 chunk）；其他 = off。**无中间值可调。**

| 配置 | rep1（含预热） | rep2 | rep3 | 稳态中位 |
|---|---:|---:|---:|---:|
| unset（默认） | 2055.58 | 2085.77 | 2084.71 | **2085.24** |
| `=1` | 1906.89 | 2029.84 | 2062.67 | **2046.26** |

**−1.9%，否决。** 口径：`cold_prompt.py`，10751–10754 token，未缓存 100%，`max_tokens=1`。

**机制**：10752 token 约 5376 压缩键（ratio 2），远低于 65536。**短 chunk 切分的 all-gather 开销大于省下的计算**——默认阈值是设对的。长 prompt（≥131072 token）默认已自动启用，无需干预。

**arena 3766 MiB 不变**，实证 patch 0001 零内存增量（patch 内无任何新分配，只划分行区间复用现有缓冲）。

**已退回 unset**，local.sh 留否决注释。

---

## 4. 步骤3a 第一步已执行：最小集试落 v0.7.5（结果乐观）

在 `/tmp/tf-v075`（`e100ee8b` 的 git worktree，**scratch，不碰生产**）试落 6 个承重 patch。

**直接 `git apply --check`：只有 0025 能落，5 个失败。** 但改用 **`--3way`**（按 blob 祖先三方合并）后结果大幅改善：

| Patch | 干净文件 | 冲突文件 | 是否落地 |
|---|---:|---:|---|
| `0008-pool-window` | 1 | 2 | ✅ 落地（`pool.zig` 新建、`TF_DS_POOL` 在位） |
| `0011-cuda-budget-parked-pages` | 0 | 2 | ✅ 落地（`cuda.zig` 含 parked） |
| `0012-dsv41-base-fixes` | 0 | 2 | ✅ 落地 |
| **`0024-mixed-pack`** | **3** | **0** | ✅ **完全干净** |
| **`0025-widen-bf16`** | **1** | **0** | ✅ **完全干净**（`widenBFloat` 在位） |
| `0026-plain-rows` | 0 | 3 | ⏸ 被挡（见下） |

**实际按序落地后的冲突总量：8 个冲突块，分布 4 个文件**

```
zig/src/native/cuda.zig          3
zig/src/native/cuda_memory.zig   1
zig/src/families/dsv41/model.zig 3
zig/src/families/dsv41/native.zig 1
```

**`0026` 未落地的原因是被前面未合并的文件挡住**，不是它自身不兼容：
```
error: zig/src/families/dsv41/model.zig: does not exist in index   (UU 未合并态)
error: zig/src/families/dsv41/native.zig: does not exist in index
Applied patch to 'zig/src/families/dsv41/prompt.zig' cleanly.      ← 它自己的 hunk 是干净的
```
⇒ **解完那 8 个冲突并 `git add` 后，0026 应可直接落。**

### 工时判断（据实测，非估算）

**最难的格式关卡 `0024-mixed-pack` 三方合并完全干净**——这是本轮最有价值的发现。EAQ v1.1 的 27 个 BF16 linear 路径可以几乎零成本搬到 v0.7.5。

**3a 的真实工作量 = 手工解 8 个冲突块（4 文件）+ 落 0026 + 构建验证。** 这与先前"50 个 patch 全失败、`lane_host.zig` 679 行 diff"的悲观印象**不是一个量级**——因为那是把**全部 52 个**往 tp2 tip 上硬套；而 3a 只要 6 个承重 patch 落到**验证过的 v0.7.5**，且用三方合并而非直接 apply。

**⚠ 尚未做**：冲突未解、未构建、未运行。以上只是"可落地性"证据，**不是"能跑"的证据**。

---

## 5. 尚未回答

1. `0010-rdma-idle-nap` 是否已被 e100ee8b 包含——需核实
2. 8 个冲突块的具体内容未逐一审读，解法未定
3. 最小集构建后能否加载我方 checkpoint 并产出正确 token——**未验证**，这是 3a 的真正验收
4. B+C 类（KV分片/spill/内存安全）往 v0.7.5 的可落地性未试，3b 规模未知
5. D 类 17 个 API patch 的重端口工时未估
6. 构建耗时：`prepare.sh` 注释「two Zig builds, ~6 min on a Spark」，按纪律 >1min 须先报时长获确认