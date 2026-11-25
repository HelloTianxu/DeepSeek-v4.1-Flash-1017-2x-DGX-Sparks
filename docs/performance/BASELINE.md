# 性能优化基线版本（Performance Baseline）

> **⚠️ 本页是 `perf-baseline-20261124` 的历史快照，已被取代为当前口径。**
> **当前基线是 `base-alias`，唯一权威口径见 [CURRENT-BASELINE.md](<../CURRENT-BASELINE.md>)。**
> 本页所说「唯一的功能性改动是 ENGRAM_AIO 1→0」**已不完整**：此后又并入
> **patch 0068 `TF_DS_PROMPT_ALIAS=1`**（arena 3766→2208 MiB，位一致 31/31），
> 生产镜像已换为 `tensorfold-dsv41:base-alias`。原文保留不改。

**基线标识** `perf-baseline-20261124`　**确立日期** 2026-11-24（Asia/Shanghai）
**状态** 已落地、已重启验证、已入库。**此后所有重启都以本基线配置为准。**

---

## 1. 这个基线改了什么

**唯一的功能性改动：`ENGRAM_AIO` 默认值 `1` → `0`**（`scripts/config.sh`）。

Engram 表的读取从 **Linux AIO + O_DIRECT（单提交线程）** 改为 **reader pool（多线程 + 页缓存）**。

**没有改**：模型、量化、任何数值内核、patch、镜像、KV/pool/context、内存守卫、认证、spill。
**输出比特不受影响**——只改 I/O 供给路径，不改数学执行。

| 项 | 基线值 | 上游默认 |
|---|---|---|
| `ENGRAM_AIO` | **0** | 1 |

其余配置沿用原有：TP2、RDMA=1、EAQ v1.1 2.9bpw + abliterated overlay、
`CONTEXT=524288`、`POOL=1052672`、`SPILL_GIB=64`、`KV_SHARD=1`、
`GRAPHS=1`、`GRAPH_CAP=128`、`SIDE=1`、`PREFETCH=1`、`PREFILL_INTERLEAVE_MS=0`、
`MEMGUARD_GIB=2.5`、镜像 `tensorfold-dsv41:exl3-tp2-8348338a8d90`、
engine pin `fbbc97132cd004237d9887b8d3185896efaedc5c`。

---

## 2. 基线性能（本机实测，非上游历史表）

口径：**实际未缓存输入 token 数 / `server.prefill_seconds`**，cached 全为 0，
自然语料冷前缀，`max_tokens=1`、thinking off。**不与 README 的历史表作 A/B 比较。**

| 负载 | 上游 `AIO=1` | **本基线 `AIO=0`** | 提升 |
|---|---:|---:|---:|
| 约 8K（10 750–10 755 tok） | 1773.36（区间 1547.19–1789.16） | **2118.93**（稳态 n=6 中位 **2118.65**） | **+19.5%** |
| 约 41K（41 364–41 368 tok） | 1833.80 | **2202.84** | **+20.1%** |

**判据是分布不重叠**：基线最慢的一次 **1955.22** 仍快过上游最快的一次 **1789.16**。
**抖动从 15.6% 降到 0.9%。**

另测 `AIO=1 + TF_DS_ENGRAM_LEAD=1` 组合：8K 中位 1903.46、41K 2137.23，**不如本基线**。

---

## 3. 为什么快（机制）

1. Engram 供给每 chunk 每层 224–249 ms，其中 hash 仅 0.5 ms、decode 仅 1.7 ms，**其余全是 `aio_gather`**。
2. 再拆：**`start_ms` 225–237 ms（提交）vs `wait_ms` 5–12 ms（等完成）** ⇒ **卡在提交，不是等盘**。
3. 读放大 **31.03×**：每 rank 每层每 chunk 有用 6.19 MiB，却发 49 152 次 4 KiB 读 = 192 MiB。
4. 盘 YMTC PC450 有 **16 个硬件队列 × 1023 tags**，但单提交线程每批 2048 个读经 blk-mq
   **全落到同一个队列**（2.002× 超 tags）⇒ `io_submit` 阻塞等 tag 回收。既有
   `blk_mq_get_tag` 睡眠栈与之吻合。
5. reader pool 多线程把读**打散到 16 个队列**，并让页缓存对重复 4 KiB 块去重。

**这不是盘到顶了，是提交路径只用了 1/16 的队列容量。**

---

## 4. 重启与验收

### 4.1 正常重启（基线自动生效）

```bash
ssh ztxjzh@192.168.1.118 'systemctl --user restart dsv41-tensorfold.service'
```

`ENGRAM_AIO=0` 已是 `scripts/config.sh` 的默认值，**无需任何本地覆盖**。
（`scripts/local.sh` 被 gitignore，是机器本地配置；基线不依赖它。）

### 4.2 验收三条

```bash
# 1. 健康
curl -s http://127.0.0.1:8888/health          # {"status": "ok"}
# 2. 两 rank 都生效
docker exec dsv41-1017-exl3-tf printenv TF_DS_ENGRAM_AIO                    # 0
ssh ztxjzh@192.168.100.11 'docker inspect dsv41-1017-exl3-tf --format "{{.Args}}"' | grep -A1 -- --aio   # 0
# 3. 冷前缀速度应落在 2100–2125 tok/s（约 8K）
```

**已于 2026-11-24 按此验证通过**：移除 `local.sh` 中的临时覆盖后重启，
rank0 `AIO=0`、rank1 `--aio 0`，实测 2100.93 / 2099.87 tok/s，
内存最低 8.262 GiB，`/health` ok。**证明基线由仓库配置自身保证。**

> 注：**每次重启后的第一个请求偏低**（约 1940–1955 tok/s），第二个起回到稳态。
> 这是重启预热，不是回退，已在多轮中复现。

---

## 5. 回滚

```bash
# 方式一：临时回到上游路径（不改仓库）
ssh ztxjzh@192.168.1.118 'cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks && \
  echo "ENGRAM_AIO=1" >> scripts/local.sh && \
  systemctl --user restart dsv41-tensorfold.service'

# 方式二：恢复部署副本的原始 config.sh
ssh ztxjzh@192.168.1.118 'cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks && \
  cp scripts/config.sh.pre-baseline-bak scripts/config.sh && \
  systemctl --user restart dsv41-tensorfold.service'
```

**回滚不涉及构建**，无镜像/patch/模型变更。

---

## 6. 内存安全边界（必读）

- **生产多路上下文、KV 累积时，head 最小空闲约 4.8 GiB**（用户实测）。
  空载单流读到的 8.2–8.7 GiB **不代表生产**。
- 本基线**不增加内存占用**：实测期间最低点 8.262–8.314 GiB，**高于上游 AIO=1 的 8.199**。
  reader pool 的页缓存是可回收内存，计入 `MemAvailable`。
- **但生产多路态未复验**。上线后应观察 head `MemAvailable` 是否仍守在 4.8 GiB 一线。
- 完整硬约束见 [发现与结论 §0](01-发现与结论.md)：**驻留态禁 kernel replay、实验容器必须带内存硬上限、
  不得绕过 memguard、不得下调 `TENSORFOLD_MEMORY_RESERVE_GIB`**。违反会要求人工拔电源。

---

## 6b. 卡点已彻底疏解（直接判决，非仅改善）

端到端变快不等于卡点消失。用零注入 GPU PM 采样、**不依赖分段阈值**的口径
（直接数活跃窗口内 `GR Engine Active ≤ 20%` 的采样点）同法对照：

| | 活跃窗口 | **空闲占比** | **>10 ms 空闲段** | 最长段 |
|---|---:|---:|---:|---:|
| 上游 `AIO=1` | 6.3754 s | **17.55%**（1.1187 s） | **7 个，合计 1.0975 s** | 218.82 ms |
| **本基线 `AIO=0`** | 5.1678 s | **0.04%**（0.0020 s） | **0 个** | **0.2 ms** |

**本基线下 prefill 期间 GPU 繁忙度 99.96%，Engram 不再是性能卡点。**
旧的 7 段长空闲（166–219 ms）全部消失；剩余 10 个空闲采样均为孤立单点（各 0.2 ms）。

**省的是等待不是工作**：prefill 省的 1.2076 s 中 1.1167 s（92.5%）来自空隙消除，
真实 GPU 工作量仅 −0.8%。Engram 的读改为**与 GPU 工作重叠**，不是消失。

**含义**：空闲已榨干，后续只能减少真实 GPU 工作量或提高内核效率。
边界：PM 采样下、单请求单流空载；生产多路未复验。
[判决全文 §9](<../../runs/forge-20261124-engram-aio/REPORT.md>)。

## 7. 本基线**未**达成与已否决的方向

- **4000 输入 tok/s 目标未达成。** 本基线约 2119（8K）/ 2203（41K）。
- **加大 chunk 已双重否决**：
  ①**Amdahl**——专家 core 归零上限仅 **2815.83 tok/s**（**⛔ 旧口径，2026-11-25 审计更正：本值基于 8628 tok/4972.654ms 旧窗口，两重污染。当前口径对应值为 3598.46 tok/s，见 CURRENT-BASELINE §0**），专家 core + 全部间隙都归零仍剩 2.393 s ≈ 3605 tok/s，**均低于 4000**；
  ②**内存**——4096 档 arena 增量 1.31–3.68 GiB，而生产可用余量仅约 2.3 GiB。
  另 `POOL=1052672` 对 4096/8192 不整除会 `BadPoolSize` 拒绝启动。
- 下一个零内存成本的目标：**collective 0.539 s（10.84%）**，其中传输与 peer 等待尚未分离。

---

## 8. 证据

- [E-ENGRAM-01 完整报告与 A/B/C 原始数据](<../../runs/forge-20261124-engram-aio/REPORT.md>)（19 份请求 JSON + 7 份内存序列）
- [E-CHUNK-MEM 内存测算](<../../runs/forge-20261124-chunk-mem/REPORT.md>)
- [E-DRAM-01 带宽缺口分离](<../../runs/forge-20261123-dram/REPORT.md>)
- [E-ROOF-01 硬件屋顶](<../../runs/forge-20261123-roofline/REPORT.md>)
- [时间账与 Amdahl 边界](<../../runs/forge-20261123-bottleneck-ledger/REPORT.md>)