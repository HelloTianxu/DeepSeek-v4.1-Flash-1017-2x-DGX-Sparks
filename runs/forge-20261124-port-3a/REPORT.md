# E-PORT-3a · v0.7.5 + 承重最小集：构建成功、加载成功、推理正确

**日期**：2026-11-24
**结论**：**3a 的核心目标达成** —— v0.7.5 引擎能加载我方 EAQ v1.1 checkpoint 并产出正确结果。
**但性能对比尚不成立**（两个性能关键特性被迫关闭，见 §4）。
**生产已恢复原配置**：kit fdde3285 / PARALLEL=4 / ROUND_ROWS=16 / KV_SHARD=1 / arena 3766 MiB / avail 8G。

---

## 0. 一句话

**「社区最佳 v0.7.5 + 我方 4 个承重 patch」这条路走通了**，最小集比预想小（4 个而非 6 个），
冲突比预想少（7 块而非几十处），最难的格式关卡 `0024-mixed-pack` 三方合并**完全干净**。

---

## 1. 最小集：从 6 个缩到 4 个

| Patch | 纳入 | 理由 |
|---|:--:|---|
| `0011-cuda-budget-parked-pages` | ✅ | 启动预算计入 per-CPU parked pages，**内存安全** |
| `0024-mixed-pack` | ✅ | EAQ v1.1 的 27 个 BF16 linear 走 cuBLAS，**格式承重** |
| `0025-widen-bf16` | ✅ | BF16 源精确展宽 f32，**格式承重** |
| `0026-plain-rows` | ✅ | plain linear 定形保 token bits，**正确性承重** |
| ~~`0008-pool-window`~~ | ❌ | **v0.7.5 原生等价**：`--pool` / `--context` / `TF_DS_WINDOW`，`ctx_cap = min(context, pool)` |
| ~~`0012-dsv41-base-fixes`~~ | ❌ | 一半是未纳入的 0010 的 Linux prctl 配套，另一半 Engram 修复 3a 不需要 |

**0008 被丢弃连带消掉 `pool.zig` 与其在 model/native 的冲突**，这是本轮工作量下降的主因。

### ⚠ 一个必须记住的语义陷阱

**v0.7.5 完全不读 `TF_DS_POOL`**（0 命中）。它的 `pool = TF_DS_WINDOW 或 --context`。
而我方 `start.sh:286` 导出 `TF_DS_POOL=$POOL` **和** `TF_DS_WINDOW=$CONTEXT`——
**我方 WINDOW 是「单请求窗口」，上游 WINDOW 是「共享池」，含义相反。**

⇒ 跑 v0.7.5 时 **POOL 必须留空**，否则两 rank 池不对称（rank0 得 524288、rank1 得 1052672）。
留空后两 rank 同为 `--context` 值，对称且更省内存。

---

## 2. 冲突解决（7 块 / 4 文件）

| 文件 | 块数 | 解法 |
|---|---:|---|
| `native/cuda_memory.zig` | 1 | `procText`（上游）与 parked 三件套（我方）是**不同函数，并存** |
| `native/cuda.zig` | 3 | ① `procText` 已移入 budget 模块 → 改限定名；② **保留上游更新的准入逻辑**（`learn_bytes`/`peer_room`/`free_streams`），只加回 parked 日志；③ 测试区并存，但**删除重复的 `chipClass` 测试**（上游已移至 `cuda_device.zig`，逐字相同） |
| `families/dsv41/model.zig` | 1 | **只取 `m.eng.plain_rows = o.plain_rows`**，丢弃 theirs 侧混入的 KV_SHARD 代码（属 3b） |
| `families/dsv41/native.zig` | 2 | 保留上游 `streams=32`/`16..64` 文档与错误串，追加 `PLAIN_ROWS`/`BadPlainRows` |

**原则**：上游演进过的地方**取上游**，我方独有的功能**加回去**。不是简单二选一。

### 构建后修的两个编译错误

- `Pool` → `budget.Pool`（上游改了引用方式）
- 删除重复 `chipClass` 测试（`use of undeclared identifier`，因函数已随上游迁走）

---

## 3. 验证链

| 阶段 | 结果 |
|---|---|
| Mac 主机测试（zig **0.17.0**） | **426/433 通过**，我方改动全部编译通过 |
| Mac 剩余失败归因 | `link.zig` 的 `SO_BUSY_POLL` 为 **Linux 专有**；该文件 `git diff` 为空、4 个 patch 均不含该符号 ⇒ **非本次引入**。另 3 个 benchmark 信号/计时测试 |
| Spark 构建（Linux aarch64 / SM121 / -j16 / 40g 内存上限） | **BUILD_OK**，产出 `tensorfold-native` 45M、`tf-dsv41-lanes` 14M |
| 加载我方 checkpoint | ✅ **`info: mixed pack: 27 float linears`** |
| rank cache | ✅ `95.1 GiB of expert entries read whole in 20.9 s`（生产 20.6 s） |
| 推理正确性（RDMA=1） | ❌ `error.Failed` |
| 推理正确性（RDMA=0 / NCCL） | ✅ 回答 "7"，`reasoning_content` 正常 |

**工具链注意**：引擎 `build.zig:25` 硬性 `@compileError` 要求 **恰好 0.17.0**；Mac 原装 0.17.1 不满足，
已下 0.17.0 到 `/tmp/zig0170/`（**重启会丢，需重下**）。

---

## 4. ⚠ 性能数字**不可用于对比**

被迫关闭两个性能关键特性：

| 特性 | 状态 | 原因 |
|---|---|---|
| **CUDA graphs** | **GRAPHS=0** | 上游**无 `--graph-cap`**（那是我方 0033 加的），GRAPHS=1 会传未知 flag 拒启；且无 0033 则 graph cache 无界增长，**有内存风险** |
| **RDMA ring** | **RDMA=0（NCCL）** | v0.7.5 的 RDMA 路径失败，见 §5 |

在此**双重让分**下测得：

| 指标 | v0.7.5+4patch（NCCL / 无 graph） | 我方基线（RDMA / graph） | 差 |
|---|---:|---:|---:|
| decode N=1 | **52.99** | 59.75 | −11.3% |
| decode N=4 聚合 | **102.16**（每流 25.54） | 109.23 | −6.5% |
| prefill（10752 tok） | **2046.5** | 2085.2 | −1.9% |

**这些差值不能归因于引擎版本**——两个已知减速因素未剥离。
**要得到可信对比，必须先端口 0033（graph-cap）并修好 RDMA。**

MTP 仍工作：N=1 `tok_per_round` 2.0157、接受率 0.56；N=4 `tok_per_round` 1.93、接受率 0.66。

---

## 5. RDMA 失败：已定位到函数，未定位到根因

`rdma.zig:402 / :433`：
```zig
pub fn gather(...) Error!void {
    if (r.failure() != null) return error.Failed;
```
⇒ 是 ring **已经失败**后的返回，**根因在更早的异步失败**，日志未打印细节。

已排除：v0.7.5 **不读** `TF_RDMA_IDLE_S` / `TF_RDMA_NAP_US`（0 命中），我方这两个变量被忽略，不是它们的问题。

**候选**：0010（rdma-idle-nap，TensorFold PR #27 的 Zig 移植）、0012（触及 rdma.zig）未端口。
**未验证**，是下一步的第一个排查点。

---

## 6. 一个真实的内存发现（与性能混淆无关）

| | 我方旧引擎 | v0.7.5 |
|---|---|---|
| prompt pass 行宽 | **2048** | **4096**（默认） |
| rank1 arena | 3766 MiB | **5.91 GiB**（6054 MiB） |
| **每行成本** | **1.84 MiB/row** | **1.48 MiB/row** |
| rank0 prompt cache | — | 0.5 GiB |
| model | 104.67 GiB | 106.51 GiB |

**v0.7.5 每行内存效率高约 20%**，且**默认就跑 4096 行**——
这正是本轮最初「chunk 4096 开不起（缺 1.10 GiB）」那个问题的方向。
上游 `3b902fd21 TF_DS_PROMPT_ALIAS`（scratch 五区共享）很可能是原因，**未核实**。

⚠ 但注意：本次 v0.7.5 是 **POOL 留空（524288）+ 无 KV 分片 + 无 spill** 的配置，
与生产（POOL 1052672 + 分片 + spill 64G）**不是同一内存口径**，上表每行成本可比，总量不可直接比。

---

## 7. 【已追加完成】两个阻塞项均已修复，公平对比已做

### 7.1 RDMA 根因：**上游硬编码 path_mtu=4096，链路只有 1024**

排查路径：`grep error.Failed` → `rdma.zig:402/433` gather() → 注释「watchdog lost a peer」
→ 失败源全是 verbs 级 → **对比 pin 与 v0.7.5 的 rdma.zig diff（45增87删）** → 找到关键行。

**v0.7.5 把 `path_mtu` 写死成 `mtu_4096`(5)，我方 pin 原本是 `pathMtu(local,remote)` 协商。**
实测两个 RoCE 口：`max_mtu 4096(5)` 但 **`active_mtu 1024(3)`** ⇒ QP 按超链路能力的 MTU 配置而失败。

**解法**：把 MTU 协商回port（7 处改动）。**未选改网口 MTU**：重启会丢，且我方实测过
「1024→4096 带宽不变、端到端 −0.20%」无收益。

**排除**：0010 是省电优化非正确性修复；0012 对 rdma 只是给 0010 加 Linux prctl 守卫；
v0.7.5 不读 `TF_RDMA_IDLE_S`/`NAP_US`。**三者都不是根因**——先前把它们列为候选是错的。

### 7.2 0033 graph-cap 已端口

3 个冲突（round/native/lanes_main），新建 `graph_cache.zig`。
theirs 侧混入的 `shard_ring`/`spill_tier` 已剔除（属 3b）。

### 7.3 公平对比（RDMA=1 / GRAPHS=1 全开）

| 指标 | v0.7.5+5patch+MTU修复 | 我方基线 | 差 |
|---|---:|---:|---:|
| decode N=1 | **60.261** | 59.746 | **+0.9%** |
| decode N=4 聚合 | **109.803**（每流27.451） | 109.226（27.306） | **+0.5%** |
| prefill | **2087.81** | 2085.24 | **+0.1%** |

**性能持平，差值在噪声内。** 且 v0.7.5 默认 **4096 行 pass**（我方 2048）。

**§4 那些 −11.3% / −6.5% 的差值已被证伪**——全部来自 RDMA/graph 两项让分，不是引擎版本差异。

**条件说明**：v0.7.5 侧 `POOL留空 / KV_SHARD=0 / SPILL=0`，我方基线 `POOL=1052672 / 分片 / spill64G`。
**是「最小集 vs 完全体」，在此不利条件下仍持平。**

### 7.4 剩余下一步
1. 3b：KV 分片五件套 + spill 四件套（**注意 0008 已丢弃，这些原本建立在 0008 的 pool 上，需改造到上游原生 pool**）
2. C 类其余内存安全 patch（0023 prepare 地板等）
3. D 类 17 个 API patch

**部署知识底稿**：`docs/DEPLOY-v075-NOTES.md` —— 8 个坑按「症状→根因→解法→验证」记录，
含脚本骨架、冲突解决原则、内存纪律、回滚。供一键部署脚本构造。

---

## 8. 产物与回滚

**Git**：`zig/patches-v075/{0001-load-bearing-4-on-v075.patch,README.md}`（commit `23abece`）
**原始数据**：`raw/pf-{base,idx,v075}-{1,2,3}.json`（9 份）

**Spark 侧（不在 git）**：
- `~/build-v075/`（源码树）、`~/build-v075-out/`（二进制）、`~/build-v075.log`
- `~/img-v075/`（派生镜像上下文，两节点都有）
- 镜像 `tensorfold-dsv41:v075-lb4`（两节点）
- 备份 `scripts/local.sh.pre-3a-bak`

**回滚**：已执行。`cp scripts/local.sh.pre-3a-bak scripts/local.sh` + systemctl 重启，
验证 arena 3766 MiB / 4 streams / KV_SHARD=1 / avail 8G，与基线一致。

**全程内存纪律**：构建前停服腾出 118 GiB；构建容器 `--memory=40g --memory-swap=40g`；
绝不并行两引擎；每次启动轮询 health 并读 MemAvailable。**未发生内存事故。**