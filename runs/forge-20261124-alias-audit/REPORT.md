# E-ABS-01 · `TF_DS_PROMPT_ALIAS` 吸收前审计

**社区坐标**：`3b902fd21` — *dsv41: TF_DS_PROMPT_ALIAS: the prompt pass's large scratch in five shared regions (TP2)*
**审计对象**：该特性能否安全吸收进我方基线（KV 分片常开）
**结论**：**可以吸收，风险低**。分片路径不参与任何共享缓冲的跨 rank 交换。
但必须用位一致性验收，且 blocked-union 路径需单独验。

---

## 1. 它做什么 / 为什么 / 怎么做

- **做什么**：`TF_DS_PROMPT_ALIAS=1` 时，`Chunk.init` 把 prompt pass 的 25 块 scratch
  归并进 **5 个共享区**，每区按**最大成员**定容，替代各成员单独 `a.take`。
- **为什么**：2048 行 / 1M 上下文下每 rank 省约 **1.6 GiB**（chunk ~2973 → ~1343 MiB），
  腾出的内存可给更大的 pool。
- **怎么做**：`Regions.take()` 只登记成员指针与区号、延迟分配；
  `Regions.place()` 每区取一次，再把所有成员指向区首。
- **正确性论证（作者的）**：同区成员**从不同时存活**，且每个成员在被读之前
  由它自己的 launch **整块写过**。作者在 world 2 下对每块缓冲做了首写/末读映射并对照代码核验。

### 1.1 五区成员清单（从 diff 逐条提取）

| 区 | 成员 |
|---|---|
| 0 | `ga`, `xf`, `gm`, `score`, `pbufs.cand`, `ekg`, `ws.w` |
| 1 | `keys`, `pool`, `ek`, `taps` |
| 2 | `o`, `pa`, `pm`, `iq`, `pidx`, `kv`, `xs.xg`, `xs.xd` |
| 3 | `q`, `u`, `xs.xu` |
| 4 | `iq4`, `ws.xh` |

**明确不共享**（作者列出，已在代码核实为普通 `a.take`）：
`h`, `h_alt`, `x`, **`cidx`**, experts 分组态(`ids`/`count`/`counts`/`members`), `xsd`,
cuBLAS workspace, `gxh`, `gz`, 计数器, `neg`, `ws.h`。

---

## 2. 两道启用门槛：我方均通过（实测核对）

代码中的 guard：
```zig
if (e.two != null) return notPorted(@src());
for (e.w.layers, 0..) |lay, li| {
    if (lay.engram_wkv != null and li >= c.candidate_source) return error.PromptAliasLayers;
    if (lay.idx_wq_b != null or lay.engram_wkv != null) last = li;
}
for (c.dspark_taps.slice()) |li| if (li <= last) return error.PromptAliasLayers;
```

我方 checkpoint（`aee640e2-ablit-0bdc2036/checkpoint/config.json` + safetensors index）实测：

| 量 | 我方值 | 判定 |
|---|---|---|
| `e.two`（四节点 2D split 节点） | 我方 TP2 → `m.two = null` | ✅ 通过 |
| `candidate_source_layer_id` | 20 | — |
| `engram_layer_ids` | [1, 14]，均 < 20 | ✅ 通过 |
| indexer 层（`attn.indexer.wq_b`） | [2, 8, 14, 20, 24, 28, 32, 36] | — |
| `last` = max(indexer ∪ engram) | **36** | — |
| `dspark_target_layer_ids`（taps） | [37, 38, 39]，全 > 36 | ✅ 通过 |

**该特性在我方 checkpoint 上能真正启用，不会返回 `PromptAliasLayers`。**

---

## 3. 核心审计：分片路径是否破坏「同区成员不同时存活」

### 3.1 审计方法

不采信作者在上游（**无 KV 分片**）上的证明，直接查我方分片路径实际触碰哪些缓冲：
读 `prompt.zig` 的 `shardAttend` / `unionOf` 与 `Attn` 构造点，逐个比对 §1.1 清单。

### 3.2 证据

**注意力的输入输出确实落在共享区**（`prompt.zig:998-999`）：
```zig
.q   = ch.q,   // 区 3
.out = ch.o,   // 区 2
```

**分片 fetch 的交换缓冲全部独立分配**（`prompt.zig:501-506`，均为普通 `a.take`，**不在 alias 清单**）：
```zig
ch.sf_codes  = try a.take(fetch_rows * c.head_dim / 2);
ch.sf_scales = try a.take(fetch_rows * c.head_dim / 16);
ch.sf_idx    = try a.take(ch.cap * (c.index_topk + 16) * 8);
ch.sf_cum    = try a.take(ch.sf_cum_rows * 4);
ch.sf_sums   = try a.take((shard_ops.scan_segs + 1) * 4);
```

**`ch.cidx`（`prompt.zig:382`）也是普通 `a.take`**，与作者「Never shared」清单一致。

### 3.3 逐路径判定

| 分片路径 | 跨 rank 交换的源 / 宿 | 是否触碰共享区 |
|---|---|---|
| contiguous · owner 发 | 源 = `a.comp.fp4.codes/scales`（**Caches 平面**，非 chunk scratch） | ❌ 否 |
| contiguous · 非 owner 收 | 宿 = `ch.sf_codes` / `ch.sf_scales`（独立） | ❌ 否 |
| union · owner 发 | `unionCompact` 读 Caches 平面 → 写 `sf_codes/scales` → 发送 | ❌ 否 |
| union · 非 owner 收 | `unionRenumber` 读 **`ch.cidx`**（不共享）→ 写 `sf_idx`（独立）；收进 `sf_codes/scales` | ❌ 否 |
| `unionOf` 计数 | 写 `sf_cum` / `sf_sums`（独立），读 `cidx`（不共享） | ❌ 否 |
| borrowed twins | `own = null`，视图退化复制，**整个 fetch 被跳过** | ❌ 否 |

**结论一：没有任何共享区成员充当跨 rank 发送源或接收宿。**
共享区只承载本 rank 的注意力/专家计算，交换一律走独立的 `sf_*` 缓冲。

### 3.4 分块循环的存活重叠检查

union 路径在 `B != n` 时分块多次调用 `sparseAttn`：
```zig
b.q = a.q + blk.act;  b.out = a.out + blk.act;  b.rows = blk.m;
```
即对 `ch.q`（区3）/ `ch.o`（区2）的**不同切片**重复读写。块间只做
`unionOf`（写 `sf_cum`/`sf_sums`）、`unionRenumber`（写 `sf_idx`）、send/recv（`sf_codes/scales`）——
**全部不在共享区**，因此不会在块间覆盖同区其他成员，也不会破坏已写入的 `ch.o` 前序块。

同区其他成员的时序也未被分片改变：
`u`（区3）、`pa`（区2）在注意力**之后**的 `wo_b` GEMM 才使用（`prompt.zig:1053`），
与 `q`/`o` 顺序存活，与上游复制路径一致。

**结论二：分片没有引入新的同区存活重叠，也没有改变层内缓冲时序。**

---

## 4. 残余风险（如实列出，不掩盖）

1. **两侧证明都是代码级，不是运行时证明。**
   作者做的是首写/末读映射对照代码；本审计同样是代码级。
   **因此不能仅凭本审计就宣布安全**，必须做位一致性实测验收。
2. **blocked-union 路径本身在我方就标注未验**：
   `prompt.zig:1067` 原文 — *"Needs GPU validation of the blocked launches against the whole-chunk one"*。
   这是分片自带的既有缺口，与 alias 无关，但 **alias + blocked-union 是最少被验证的组合**，需单独覆盖。
3. **省下的内存没有已证实的速度出口**：r2 实测 4096 行 pass 的 prefill 2087.81
   vs 2048 行 2086.70，**无增益**。所以 alias 单独吸收**不会带来吞吐提升**，
   它的价值在于**解锁** `TF_DS_PASS_FROM`（8192 以上跑 4096 行）所需的内存。

---

## 5. 判决与吸收方案

**判决：收下，但按「耦合两步」执行，不单独计价。**

| 步骤 | 动作 | 验收闸门 |
|---|---|---|
| A | cherry-pick `3b902fd21`（107 行，`prompt.zig` + 2 处一行改动），开 `TF_DS_PROMPT_ALIAS=1` | ①启动日志出现共享区 info 行 ②**arena 显著下降**且 avail 上升 ③`tools/exact.py` 对 alias-off **位一致** |
| B | 在 A 腾出的内存上吸收 `TF_DS_PASS_FROM`（`3befde133` + `37b7d2651`），令 8192 以上用 4096 行 | ①**avail 守住 6 GiB 以上**（红线 4.8）②prefill 同场 A/B 有正收益，预期量级参照上游 8K +5.6% / 128K +4.5% |

**任一步闸门不过就退回该步，不影响已抬高的地板。**

**A 步单独不计收益**——它是容量改动。若 A 后 arena 未如预期下降，
说明我方 0.5M 上下文下共享区收益过小，则整条路径否决，不做 B。

---

## 6. 尚未执行

本报告**只是审计**，未 cherry-pick、未构建、未运行。
按开发手册 §2，下一步是执行 A 步并按 §3 做同场 A/B。
---

## 7. 【A 步执行结果】容量收益已实测确认

### 7.1 吸收过程

cherry-pick `3b902fd21` 到基线树 `/tmp/tf-base`（基 `d7ca30aa8`），**4 个冲突块**。
其中 3 处 theirs 混入了**不属于本特性**的上游改动，我方树无对应结构，全部剔除：

| 混入项 | 来源 | 处理 |
|---|---|---|
| `prompt.splitEnv()` | 另一 commit 把 idx_split 重构成函数 | 保留我方 inline 解析，只加 `aliasEnv()` |
| `ex.width + ex.rest.width` | parity pack 特性 | 保留我方 `ex.width` |
| `ch.key_cap = ke` | 我方 Chunk 无此字段（全树 grep 仅冲突块内 1 处） | 丢弃 |
| `var xs` 局部 → `ch.xs` | 上游把 xs 移进 Chunk | 我方 Chunk 本就有 xs 字段，直写 `ch.xs` |

吸收后 commit `227aafb55`。Spark 全新构建 **BUILD_OK**，两节点镜像 `tensorfold-dsv41:base-alias`。

### 7.2 同场 A/B（**同一二进制，仅 env 不同**，排除一切构建差异）

| 指标 | alias OFF（对照） | alias ON | 差 |
|---|---:|---:|---:|
| 启动日志 | — | `25 buffers of 2458 MiB in shared regions of 900 MiB` | — |
| **arena** | **3766 MiB** | **2208 MiB** | **−1558 MiB（−41.4%）** |
| **avail（加载后/测后）** | 8 / 7 G | **10 / 9 G** | **+2 GiB** |
| N=1 decode | 59.236 | 59.265 | +0.05% |
| N=4 聚合 | 107.518 | 107.621 | +0.10% |
| prefill 中位 | 2083.16 | 2084.42 | +0.06% |

prefill 三次：OFF 2047.85 / 2083.16 / 2086.70；ON 2055.87 / 2084.42 / 2093.99。

### 7.3 判定

1. **基线重建忠实性得证**：OFF 臂 arena **3766 MiB**，与生产基线历史值**完全一致**。
2. **容量收益确认且超预期**：省 1558 MiB，高于审计时按 0.5M 上下文折半估的 ~0.8 GiB。
   实际共享比 2458 → 900 MiB（2.73×）。
3. **性能中性**：三项均在 +0.1% 噪声内，**与「容量特性不提速」的预期完全相符**（手册 §2 第 5 步）。
4. 闸门①启动日志 ✅　闸门②arena 显著下降 ✅　闸门③位一致性 ⏳ 进行中。

**A 步不计吞吐收益**，其价值是解锁 B 步（`TF_DS_PASS_FROM`，8192 以上跑 4096 行）所需的内存。
现已多出 1558 MiB，而 4096 行 pass 此前实测需 arena 6144 MiB（相对 2048 行的 3872 多 2272 MiB）——
**注意：1558 < 2272，缺口约 714 MiB，B 步是否放得下需实测，不能想当然。**

### 7.4 闸门③ 位一致性：**通过**

`tools/exact.py --streams 4`，同一二进制、仅 `TF_DS_PROMPT_ALIAS` 开关不同，
ON 先录、OFF 用 `--like` 复用同盐同形状：

```
wrote ex-alias-on.json : 31 cases, 0 errors, 0 failed checks
wrote ex-alias-off.json: 31 cases, 0 errors, 0 failed checks
RESULT: IDENTICAL (0 problem(s), 31 cases)   exit=0
```

内部检查亦全过：`solo_k == conc_k`、`resume_2/3` 缓存命中 6144、repeat 一致。
覆盖 len_1…len_100000、conv_A1..A3 交错、resume、conc_1..4 + solo_1..4。

---

## 8. 最终判决：**收下，已并入基线**

三道闸门全过（启动日志 ✅ / arena −1558 MiB ✅ / IDENTICAL 31/31 ✅），
收益方向与量级均与预期相符（容量增益、性能中性）。

- 项目 patch：**`zig/patches/0068-prompt-alias-shared-regions.patch`**（257 行）
- 序列验证：干净 pin 上按序落 **53/53 全部干净**
- `zig/patches/README.md` 已补 0068 条目（含开关、默认、实测与审计出处）
- 基线地板抬高：**arena 3766 → 2208 MiB，head 多 2 GiB 余量**

### 8.1 对 B 步的诚实修正

审计 §5 曾设想 alias 腾出的内存可解锁 4096 行 pass。**实测后须修正**：
4096 行 pass 相对 2048 行需多 2272 MiB arena，而 alias 只省出 **1558 MiB**，
**缺口约 714 MiB**。因此 B 步**不能想当然认为放得下**，
必须实测 `TF_DS_PROMPT_ALIAS=1` + 4096 行下的 arena 与 avail，
以 avail ≥ 6 GiB 为闸门决定去留。

### 8.2 尚未做

- B 步（`TF_DS_PASS_FROM`）未吸收、未实测。
- 生产多路（head 4.8 GiB 红线）下未复验，本轮为空载。
