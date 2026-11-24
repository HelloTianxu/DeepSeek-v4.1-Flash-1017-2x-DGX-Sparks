# E-KVOFF-01：KV 卸载 SSD 换 chunk 4096 —— 不成立，并更正一处 6.8× 高估

> **判决：不行。KV 池实测只有 0.52 GiB/rank，全部卸掉也补不上 chunk 4096 的 1.10 GiB 缺口。**
>
> **连带更正：此前「CONTEXT 减半释放 2.5 GiB」是反推出来的幻影，真实只有 0.386 GiB（高估 6.8×）。
> 基于它的 chunk 4096「+23.0%」条件投影，内存前提不成立，一并作废。**

全程只读（metrics / 源码 / config），未停服、未改配置，MemAvailable 7.62 GiB。

---

## 0. 结论表

| 问题 | 实测/推导 | 判定 |
|---|---|---|
| KV 池到底多大 | **532.07 MiB = 0.52 GiB/rank** | 公式与日志精确吻合 |
| 随 CONTEXT 变化的总量 | **0.770 GiB**（KV 532 MiB + RoPE 256 MiB） | 就这两项 |
| 此前声称的「KV 约 5.24 GiB」 | 实为非专家**权重**，不随 CONTEXT 变 | **高估 6.81×** |
| chunk 4096 需求 | arena 增量 **3.40 GiB** | — |
| 生产可用 | 4.8 − 2.5(memguard) = **2.3 GiB** | 缺 **1.10 GiB** |
| 卸全部 KV 后 | 2.82 GiB | **仍缺 0.58 GiB** ❌ |

---

## 1. 先肯定思路：KV 确实有冷热，引擎也已经在做

用户的直觉方向是对的——**KV 和专家权重不同，它确实有冷热之分**。
而且引擎**早就实现了**：`SPILL_GIB=64` 的 spill v2 就是把
**kept prompt states（空闲会话的 KV）**写到各自本地 NVMe，
下次读回替代重算 prefill（`PORT-RECIPE.md`：*"read back instead of a new prefill"*）。

但要分清两件事：

| | spill tier（已有） | 用户设想的「KV 卸载」 |
|---|---|---|
| 对象 | **空闲会话**的已完成 prompt 状态 | **活跃流**正在用的 KV 池 |
| 时机 | 会话结束后落盘，下次命中读回 | 每次 attention 都要读 |
| 能否在盘上 | ✅ 可以，本来就在盘上 | ❌ 活跃 KV 必须设备驻留 |

**所以「卸载 KV」这件事，能做的部分引擎已经做了；不能做的部分是物理约束。**
真正的问题不在能不能，而在**量够不够**——下面算。

---

## 2. 精确公式与验证（这是本轮最硬的证据）

源码 `shard.zig:157 poolBytes`：

```zig
for (sources) |src| {
    const full = pool / src.ratio + pad_rows;
    const comp_rows = if (sharded) pool/world/src.ratio + pad_rows + trash else full;
    n += src.comp_row * comp_rows + src.idx_row * full;
}
```

`model.zig:521 kvSources`：`comp_row = head_dim/2 + head_dim/16`，`idx_row = index_head_dim/2 + /32`。
代入本 checkpoint（`head_dim 512`、`index_head_dim 128`）⇒ **comp_row 288 B、idx_row 68 B**。

**关键结构**：压缩行按 rank 分片，**索引键是全池复制的**（`idx_row * full`，不除 world）。

config 实参：`kv_source_layer_ids = [2,8,14,20]`（**只有 4 层是 KV 源**，不是 40 层），
对应 `compress_ratios = [2,2,2,1]` ⇒ **Σ1/r = 2.5**。

| 量 | 公式算出 | 引擎日志 | 吻合 |
|---|---:|---:|---|
| sharded（每 rank） | **532.07 MiB** | 532 MiB | ✅ |
| replicated | **893.48 MiB** | 893 MiB | ✅ |

**两个数同时对上，公式与参数确认无误。**

⇒ **KV 池 = 0.52 GiB/rank。** 之所以这么小：只有 4 层留压缩 KV，
每 position 每层仅 288 B，且 `sliding_window 128` + `index_topk 512` 的稀疏注意力
本来就不需要保全量稠密 KV。**这个模型的 KV 从来不是内存大头。**

---

## 3. 更正：那 5.24 GiB 是权重，不是 KV

此前 review §5.4 用 `104.54 − 95.1 − 3.68 − 0.52 = 5.24 GiB` 反推，
把残差**整个当成 KV**，并假设它随 CONTEXT 线性缩放。review §7 自己标注了「未直接测量」。

**实测残差 5.27 GiB，但其中随 CONTEXT 变化的只有 0.770 GiB：**

| 项 | 大小 | 随 CONTEXT 变？ |
|---|---:|---|
| KV compressed planes | 532 MiB | ✅ 随 POOL |
| RoPE 表（`4×rows×32×4`） | 256 MiB | ✅ 随 WINDOW |
| **小计** | **0.770 GiB** | — |
| embed.weight（BF16 **replicated**） | 1.233 GiB | ❌ 权重 |
| attn/dense 投影 | ~1.5 GiB | ❌ 权重 |
| head.trellis（已切分） | 0.231 GiB | ❌ 权重 |
| mtp 非专家 / norm / gate / hc | ~0.4 GiB | ❌ 权重 |
| cuBLAS 工作区 / CUDA 图 / RDMA 环 / spill staging | ~1.6 GiB | ❌ 固定 |

**⇒ 高估 6.81×。**

**直接后果**：`CONTEXT 524288→262144` 实际只释放 **0.386 GiB**，不是 2.5 GiB。
生产可用 2.3 + 0.386 = **2.686 GiB < 3.40 GiB**，**chunk 4096 装不下**。
此前「2.3 + 2.5 = 4.8 > 3.68，可行，投影 +23.0%」的结论**内存前提不成立，作废**。

---

## 4. 四个方案逐个算

需求：chunk 4096 arena 增量 **3.40 GiB**（结构上界法）；生产可用 **2.3 GiB**；缺 **1.10 GiB**。

| 方案 | 释放 | 可用合计 | 装得下 4096？ | 余量 |
|---|---:|---:|---|---:|
| **A 卸载全部 KV 池**（理论上限） | 0.520 | 2.820 | ❌ | **−0.581** |
| **B CONTEXT 减半**（配置级） | 0.386 | 2.686 | ❌ | −0.715 |
| **C CONTEXT 砍到 131072**（配置级） | 0.578 | 2.878 | ❌ | −0.523 |
| **D = C + embed vocab 切分**（引擎改动） | 1.195 | 3.495 | ⚠️ 勉强 | **+0.094** |

**A 是物理上限**：把 KV 池全卸掉（实际不可能，活跃流没法算注意力），也只到 2.82 GiB。
**用户设想的路线，天花板就在缺口之下。**

**D 唯一算得过来，但不建议**：
- 余量仅 **0.094 GiB**，而 arena 估计本身有 35.7%–92% 的不确定区间，这点余量毫无安全裕度；
- 需要**同时**砍上下文到 1/4（产品能力变更）+ 改引擎给 embed 切分 + 改 patch 重编译 chunk_rows + 同改 POOL；
- 用户的硬约束是生产最小空闲 4.8 GiB、爆内存要人工拔电源。**0.09 GiB 裕度等于没有裕度。**

---

## 5. 那 chunk 4096 还有别的路吗

内存缺口 1.10 GiB 是硬缺口，已穷举的释放项加起来都不够（除非动引擎且无裕度）。
**在不改引擎的前提下，chunk 4096 方向应当关闭**，而不是继续找腾挪。

真正零内存成本的杠杆此前已落地（`ENGRAM_AIO=0`，+19.5%/+20.1%）。
decode 侧已实测到屋顶 82.8%，剩余 +20% 且不吃内存。

---

## 6. 边界与未做

- 源码 ref **f77d5a1 ≠ 生产 pin fbbc9713**。但 poolBytes 公式算出的 532.07/893.48
  与生产引擎自报 532/893 **两值同时精确吻合**，公式与参数可确信；
  RoPE 公式未在日志中单独暴露，**256 MiB 为公式推算值，未独立实测**。
- `trash` 取 0；若生产 trash>0，KV 池只会更大一点点（每源 288 B×trash 行），不改变量级结论。
- arena 增量沿用 chunk-mem 的结构上界法，**真实值不确定性无法在构建前消除**。
- **未做 CONTEXT 变更实测**（需重启，本轮未获授权）；释放量为公式推导，
  但两项公式之一已双值验证，另一项为同源码直接读取。
- 残差中 ~1.6 GiB 的固定项为归类推断，未逐项实测。

## 7. 产物

- `kv_account.py` / `kv-account.json` — 可复跑测算，含公式自校验