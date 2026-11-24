# E-KIT-01：kit v0.7.5 超集升级 + 并发/轮宽扫描 —— 16 流聚合 +79.2%

> **判决：成立并已上生产。** kit 换到 v0.7.5（严格超集）+ `PARALLEL 4→16` + `ROUND_ROWS 16→32`，
> **16 流聚合 decode 110.0 → 197.16 tok/s（+79.2%）**，单流不变（~60）。
> arena 3766 → 4000 MiB（**+234 MiB**），远低于 2.3 GiB 生产余量。
>
> **但收益全部来自并发，单人单流零收益。** 且真正的杠杆不是 kit 本身，是 kit 解锁的轮宽。

---

## 0. 结论表

| 项 | 结果 |
|---|---|
| 新 kit 兼容性 | **严格超集**：198/198 生产内核哈希全在，抽样 cubin 逐字节一致，+168 新变体 |
| 引擎/镜像 | **完全不动**（pin `fbbc9713` + 52 patch，镜像 `exl3-tp2-8348338a8d90`） |
| 是否需重编译 | **否**，kit 不在镜像内，`prepare.sh` 单独抓取、两 rank 经 NFS 读 |
| 最优配置 | **PARALLEL=16 + ROUND_ROWS=32** |
| 16 流聚合 | **197.16 tok/s**（基线 N=4 110.0，**+79.2%**） |
| 单流 N=1 | 59.83（基线 61.4，噪声内，**无收益**） |
| 内存代价 | arena **+234 MiB**，model 104.54→104.67 GiB |

---

## 1. 上游现状：我们这条分支已停更，且 kit 已封版

| 分支 | 最新提交 | 说明 |
|---|---|---|
| `deepseek-v41-zig`（我方 pin 所在） | **2026-10-24** | tip **就是** `fbbc9713`，一个月未动 |
| `deepseek-v41-zig-tp2`（活跃线） | 2026-11-23 | 与我方 pin **分叉 2220 提交**，我方独有 129 |
| kit `main` | 2026-11-19 | **我方 pin 的直系后继，仅领先 10 提交** |

**kit README 原文（维护模式声明）**：
> "This recipe is **stable at v0.7.5**. We still answer issues and review bug-fix pull requests,
> but **we are not adding new features**. Our active work is on GLM-5.3 on four DGX Sparks."

⇒ **「社区最佳状态」是一个冻结坐标**：引擎 `e100ee8b`（kit 的 `ENGINE_COMMIT` 指定）+ kit `fdde3285`。

**整体换引擎不可行**：52 个 patch 打到 tp2 上**只有 2 个能落**（18 DSV41 / 17 SERVER / 15 LANES / 2 其他）。
且 tp2 **没有** `TF_DS_KV_SHARD` 与 `SPILL_GIB`——`kv_parts.zig` 两分支皆无，是我方 patch 0018 新建。
上游 **890 B/position**，我方 **530 B/position**（KV 分片省 **40.4%**）。**换线即净亏内存。**

---

## 2. gating 验证：新 kit 吃我方 2.9bpw + ablit

上游 v0.7.0+ 围绕自家 **2.5bpw** 权重，我方是 **EXL3 EAQ v1.1 2.9bpw + ablit**，故必须先验证。

| 检查 | 结果 |
|---|---|
| aot.json 内核哈希交集 | **198 / 198**（生产 198，新 366） |
| 仅生产有 | **0** |
| 函数种类缺失 | **0** |
| 抽样 cubin sha256（`_bias`×2、`_gemm`×2） | **逐字节一致** |
| MANIFEST 383 条 | **0 不符**（Spark 侧实测） |
| `rope.json` | rows 1114112 / half 32 / 四个 sha256 **完全相同** |

⇒ **新 kit 是严格超集，零 `MissingTritonVariant` 风险。**
RoPE 表既然相同，**直接拷贝旧 kit 的 5 个生成文件即可，无需起容器跑 PyTorch**（省一次内存风险）。

---

## 3. 内存账：预测 vs 实测

源码 `round.zig:200-300`（**真 pin**，非近似树）逐项照抄 `a.take`。
`glue` 按 `max_rows=48` 固定，其余按设定 R 线性。大头 `keys` 128.5 + `score` 64.25 MiB @R=16。

| 项 | 预测 | 实测 | 判定 |
|---|---:|---:|---|
| R 16→32（4 流） | +222.89 MiB | **+152 MiB** | 预测偏保守 ✓ |
| 流 4→16（R=32） | +80 MiB | **+82 MiB** | 吻合 ✓ |
| 合计 arena | — | 3766 → **4000 MiB** | +234 MiB |

**一处差点误判**：`ringBytes = (c.window + 16) * head_dim * 2`，我先按 `c.window=524288` 算得 **+6 GiB**，
几乎否掉整个方案。核 `config.zig:177` 发现 **`c.window = sliding_window = 128`**，
真实是 144 KiB/层/流 × 40 层 = 5.625 MiB/流，4→16 仅 **+67.5 MiB**。**该 +6 GiB 结论作废。**

---

## 4. 关键机制：ROUND_ROWS 是上限不是目标

`start.sh:64` 原有硬守卫 `ROUND_ROWS == 16`，注释写明「kit 只有 16 行内核（issue #23）」。
引擎 `round.zig:36 max_rows = 48`，`205` 仅在 `R > 48` 报错 ⇒ **引擎本就接受 32，卡点纯在 kit**。
守卫已放开为 16..48 并注明依据。

**但抬高上限在低并发下完全无效**：

| PARALLEL | ROUND_ROWS | 聚合 tok/s | tok/轮 | 接受率 |
|---:|---:|---:|---:|---:|
| 4 | 16（基线） | 110.0 | — | — |
| 4 | 32 | 108.09 | 1.855 | 0.695 |
| 16 | 16 | 186.88 | **1.004** | **null** |
| **16** | **32** | **197.12 / 197.16** | 1.743 | 0.721 |
| 16 | 48 | 178.86 | 2.079 | 0.524 |

**两条机制**：
1. **4 流时每轮仅 ~7.4 行**（4×1.855），连 16 都填不满 ⇒ R=32 毫无作用（108.09 ≈ 基线）。
2. **16 流 + R=16 时 MTP 被静默关闭**：`depth()` 受 `max_rows/live−1` 限，16/16−1 = **0**，
   没有余行放 draft ⇒ tok/轮 1.004、接受率 null。R=32 时 32/16−1 = 1，投机恢复。
   **这正是 E-DECODE-01 记下的那条限制，在此被实证。**
3. **R=48 反而更慢**：depth=2 使 tok/轮升到 2.079，但接受率跌到 0.524、轮速 113→86，净亏。

**归因**：PARALLEL 4→16 贡献 **+72.9%**（108.09→186.88），ROUND_ROWS=32 再贡献 **+5.5%**（→197.12）。

---

## 5. 边界与未做

- **单人单流零收益**（59.83 vs 61.4）。收益只在多路并发兑现，对 Agent 批量场景有意义。
- 口径为 **256-token 突发**，非上游的 **90 s sustained**。上游 16 流 sustained 217.94，**不可直接对比**。
- 上游用 **2.5bpw**，我方 **2.9bpw**（字节多 16%），其单流 106.45 不能全部归因于代码。
- **生产多路 KV 下未复验**：本轮全部为空载（avail 7.8–8.2 GiB），用户生产基线是 head 最小空闲 4.8 GiB。
  arena +234 MiB 在 2.3 GiB 余量内，但**未在生产多路负载下实测**。
- 16 流共享 POOL=1052672，**长会话并发下池压力未测**（spill 兜底，但未验证）。
- prefill 未测（本轮只动 decode 相关配置，chunk 仍 2048）。
- 未推进整体换引擎到 v0.7.5（50 patch 需重端口，另立项）。

## 6. 回滚

```
cp scripts/local.sh.pre-kit-v075-bak scripts/local.sh
cp scripts/config.sh.pre-kit-v075-bak scripts/config.sh
cp start.sh.pre-kit-v075-bak start.sh
systemctl --user restart dsv41-tensorfold.service
```
冻结点：分支 `prod-fbbc9713-frozen` / tag `pre-upstream-v075-20261124`。

## 7. 产物

- `roundrows_mem.py` / `roundrows-mem.json` — 轮宽内存测算（源码逐项）
- 上游克隆于 Mac `_upstream/{tf,kit}`（**不在 Spark，不占其磁盘/内存**）