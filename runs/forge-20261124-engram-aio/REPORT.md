# Engram 供给路径 A/B：O_DIRECT 单线程提交 → reader pool

**日期** 2026-11-24　**结论** `ENGRAM_AIO=0` 落地，冷前缀 **+19.5%（8K）/ +20.1%（41K）**
**性质** 配置级单变量 A/B，**零构建、零额外内存、可一键回滚**

---

## 0. 结论先行

| 配置 | 10 753 tok 中位 | 区间 | 41 367 tok 中位 |
|---|---:|---|---:|
| **A** `ENGRAM_AIO=1`（原配置，O_DIRECT） | 1773.36 | [1547.19, 1789.16] | 1833.80 |
| **C** `AIO=1 + TF_DS_ENGRAM_LEAD=1` | 1903.46 | [1729.33, 2002.24] | 2137.23 |
| **B** `ENGRAM_AIO=0`（reader pool）**← 采纳** | **2118.93** | [1955.22, 2122.69] | **2202.84** |

**B 稳态（剔除每次重启后首请求）n=6：2102.80 / 2106.71 / 2118.38 / 2118.93 / 2120.71 / 2122.69，
中位 2118.65，跨度 0.9%。**

- **B vs A：+19.5%（8K）、+20.1%（41K）。**
- **分布不重叠**：B 的最小值 1955.22 > A 的最大值 1789.16。
- **B 同时把抖动从 15.6% 压到 0.9%**——去掉提交瓶颈后方差塌缩，是机制被命中的旁证。

---

## 1. 为什么选这条路（而不是 chunk）

`forge-20261123-bottleneck-ledger/REPORT.md` §3 的 Amdahl 边界早已给出：
专家 core 归零上限仅 **2815.83 tok/s**，专家 core + 全部间隙都归零仍剩 2.393413 s ≈ 3605 tok/s。
**chunk 杠杆的数学天花板低于 4000 目标**；且 `forge-20261124-chunk-mem` 测算表明，
在生产 head 最小空闲 4.8 GiB 的基线下，chunk 4096 的 arena 增量（1.31–3.68 GiB）装不下。

而**无 kernel 间隙 0.670695 s 占 window 13.49%**，其中 99% 集中在四个完整 chunk 的
第 1 层 Engram 入口——**是等待，不是工作，回收它不需要任何额外内存**。

---

## 2. 机制：因果已在既有证据中分离

### 2.1 时间去哪了（`engram-phase-summary.json` / `engram-aio-summary.json`）

每 chunk 每 Engram 层：

| 阶段 | 耗时 |
|---|---|
| hash | 0.515–0.671 ms |
| **aio_gather** | **224.0–248.9 ms** |
| decode | 1.495–1.851 ms |

进一步拆 aio_gather：**`start_ms` 225–237 ms（提交、24 次调用），`wait_ms` 仅 5.4–12.1 ms（等完成）**。
⇒ **时间花在提交路径上，不是等盘返回。**

### 2.2 读放大（`engram-io-account.json`）

每 rank 每层每 chunk：

- 有用字节 **6 488 064 B = 6.19 MiB**
- 最小请求字节 **201 326 592 B = 192 MiB**（49 152 次 4 KiB 读）
- **读放大 31.03×**

### 2.3 队列容量（`nvme-queue-account.json` / `nvme-queue-config.json`）

盘 **YMTC PC450-2TB-D**，ext4，`scheduler=[none]`，16 个硬件队列 × **1023 tags**。

- 每次提交 burst **2048 个读**
- **`burst_per_hwqueue_tags = 2.002`**，`capacity_shortfall_if_single_queue_no_completions = 1025`

⇒ **单提交线程的读经 blk-mq 落到同一个硬件队列，2048 > 1023 tags，
`io_submit` 必须阻塞等完成回收 tag，提交被串行化。** 49 152 读 / ~230 ms ≈ **21.4 万 IOPS**，
这是**提交路径的软件限制，不是盘的 IOPS 上限**。

### 2.4 三条候选路，实测选优

`ENGRAM_AIO=0` 走 reader pool（多线程 + 页缓存），同时打散硬件队列并对重复 4 KiB 块去重。
`TF_DS_ENGRAM_LEAD=1` 只在 AIO=1 下可用（`engram_lead.zig`，日志会打印
`engram-lead: off … it needs the Engram AIO`），为每层起专用 reader 线程并提前一个 chunk 预读。
两者都试，**B 优于 C**（8K +11.4% 对 C 中位，41K +3.1%）。

---

## 3. 方法与口径

- 主报 **实际未缓存输入 token 数 / `server.prefill_seconds`**（metrics delta），cached 全为 0。
- 探针 `cold_prompt.py`，nonce 置位置 0，`max_tokens=1`、`temperature=0`、thinking off。
- 每配置多次重复；A/B/C 各自独立重启，走 **user systemd**，不手动 `docker run`，memguard 全程在位。
- 每轮 200–300 ms 采样 head `MemAvailable`，记录最低点。
- 配置生效逐次核验：`docker exec … printenv TF_DS_ENGRAM_AIO`，C 另核验日志 `engram-lead: on`。

### 3.1 已识别的混淆与其处置

| 混淆 | 处置 |
|---|---|
| 页缓存被前一轮焐热 | A 走 O_DIRECT **不填页缓存**；B 的 41K 长 prompt 有约 3/4 内容为冷，仍立刻达 2201.63 ⇒ 主机制不是缓存焐热 |
| 重启后首请求偏低 | B-aio0 r1=1955.22、B-final r1=1943.02，**每次重启后都复现** ⇒ 判为重启预热，稳态统计中剔除并单独列出 |
| 单次噪声 | A 跨度 15.6%，故以**分布不重叠**而非均值差作判据 |

---

## 4. 内存安全

| 轮次 | head avail 起点 | **期间最低** |
|---|---:|---:|
| A-aio1 | 8.240 | 8.199 |
| B-aio0 | 8.685 | **8.275** |
| B-more | 8.324 | 8.279 |
| B-long | 8.313 | 8.266 |
| B-final | 8.634 | **8.314** |
| C-lead | 8.429 | 8.195 |
| C-long | 8.222 | 8.175 |

**B 的内存最低点始终高于 A**，未见页缓存挤压 `MemAvailable`（干净页缓存可回收，计入 MemAvailable）。
全程 memguard 地板 2.5 GiB 未触及，无重启失败、无 OOM。

**边界**：以上均为**空载单流**基线（8.2–8.7 GiB）。用户给出的**生产多路 KV 下 head 最小空闲 4.8 GiB**
未在本轮复现。reader pool 的页缓存属可回收内存，理论上不压低 MemAvailable，
但**生产态尚未实测**，上线后应观察 head `MemAvailable` 是否仍守在 4.8 GiB 一线。

---

## 5. 落地与回滚

### 5.1 已落地

Spark `~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/scripts/local.sh` 追加一行：

```
ENGRAM_AIO=0
```

备份 `scripts/local.sh.engram-ab-bak`（原配置，SHA256 见 §6）。当前服务运行于该配置，
`/health` ok，`TF_DS_ENGRAM_AIO=0` 已核验。

### 5.2 一键回滚

```bash
ssh ztxjzh@192.168.1.118 'cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks && \
  cp scripts/local.sh.engram-ab-bak scripts/local.sh && \
  systemctl --user reset-failed dsv41-tensorfold.service; \
  systemctl --user restart dsv41-tensorfold.service'
```

回到 `ENGRAM_AIO` 默认 1。**无镜像变更、无 patch 变更、无模型变更**，回滚不涉及构建。

### 5.3 验收

`curl /health` = ok；`docker exec … printenv TF_DS_ENGRAM_AIO` = 0；
冷 prompt 复测应落在 2100–2125 tok/s（8K）。

---

## 6. 边界与未做

1. **未改任何数值内核、模型、量化**；只改 I/O 供给路径，**不改数学执行**，输出比特不受影响。
2. 生产多路并发下的收益与内存未测（本轮单流）。
3. 未测 worker 侧同期内存与 Engram 时标。
4. 未直接测盘的 IOPS 上限以量化「软件限制 vs 硬件顶」的剩余空间；
   §2.3 的结论基于队列容量算术与 `start_ms`/`wait_ms` 的分离，不是 fio 实测。
5. 未做 `drop_caches` 类系统级操作（会波及 95.1 GiB rank cache，风险不对等）。
6. 4000 tok/s 目标仍未达成，且 §1 的 Amdahl 边界表明它需要同时改进多个实质工作项。

---

## 7. 产物

- `ab_reps.sh` — A/B 重复测量脚本（含内存采样与 <5 GiB 中止闸）
- `raw/` — 19 份请求 JSON + 7 份内存采样序列（只增不改）
- 本文件
---

## 8. 基线固化验证（2026-11-24，追加）

用户要求把本优化做成基线版本、后续重启都以此为准，并在仓库中体现。

**做法**：把 `ENGRAM_AIO` 的默认值从 `1` 改为 `0` 写进 **`scripts/config.sh`**（受版本控制），
而不是只留在被 gitignore 的 `scripts/local.sh` 里。随后**删除 `local.sh` 中的临时覆盖**再重启，
以证明基线由仓库配置自身保证，不依赖机器本地文件。

**验证结果**：

| 检查 | 结果 |
|---|---|
| `/health` | ok |
| rank0 `TF_DS_ENGRAM_AIO` | **0** |
| rank1 `--aio` | **0** |
| 冷前缀 r1（重启后首请求） | 1944.38 tok/s |
| 冷前缀 r2 / r3 | **2100.93 / 2099.87 tok/s** |
| 期间 head 内存最低 | 8.262 GiB |

稳态与 §0 的 B 组一致（2099.87–2122.69）。**基线成立。**

原始数据：`raw/BASELINE-r{1,2,3}.json`、`raw/mem-BASELINE.txt`。

**入库位置**：`scripts/config.sh`（默认值 + 实测依据注释）、
`docs/performance/BASELINE.md`（配置/验收/回滚/边界）、`README.md`、`CHANGELOG.md`。
部署副本备份 `scripts/config.sh.pre-baseline-bak`。标签 `perf-baseline-20261124`。
