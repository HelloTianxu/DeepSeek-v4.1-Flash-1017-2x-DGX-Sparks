# E-ENGRAM-02：ENGRAM_AIO 对 decode 的独立判决 —— 无影响，AIO=0 是纯收益

> 用户假设：**「把 Async 关掉后 prefill 1600→2000，但那之后就没好好测过 decode，现在 decode 不行了」**。
> **判决：假设否证。AIO 对 decode 的影响全部在 ±1% 噪声带内，而 prefill 差 14.77%。**
> `ENGRAM_AIO=0` **没有 decode 代价**，是纯收益，维持不变。

## 0. 为什么这个怀疑合理（必须先承认）

`docs/CURRENT-BASELINE.md:865` 原话已注明：
> 不重试 O_DIRECT（更慢）/ AIO（**`o.engram_aio` 一个开关同管 decode+prompt，历史端到端否决完整**）

而 E-ENGRAM-01（`forge-20261124-engram-aio`）的验收**只有冷前缀 prefill**：
- 结论行：「`ENGRAM_AIO=0` 落地，冷前缀 **+19.5%（8K）/ +20.1%（41K）**」
- 边界行：「单请求、单流、空载」「生产多路并发下的收益与内存未测」

**该报告 §2.1 表里的 `decode 1.495–1.851 ms` 是 Engram 表解码阶段（hash → aio_gather → decode），
不是 LLM token decode。** ⇒ **当时确实从未测过 LLM decode 侧代价**，用户指出的验证缺口真实存在。

## 1. 两臂的真实语义（更正一处常见误解）

| 臂 | 实际含义 |
|---|---|
| `AIO=1` | **O_DIRECT 单线程提交**（Linux AIO，绕过 page cache） |
| `AIO=0` | **reader pool**（线程池并发读，走 page cache） |

**不是「异步 vs 同步阻塞」。** E-ENGRAM-01 §2.1 已分离机制：
`aio_gather` 224–249 ms 中 **`start_ms` 225–237（提交路径），`wait_ms` 仅 5.4–12.1（等盘）**
⇒ reader pool 消除的是**大批量提交瓶颈**，读放大 31.03×。

## 2. A/B 实测（同场，trap 强制恢复，干预已核验容器变量）

流程：段 A0（AIO=0）→ 改 local.sh 注入 `ENGRAM_AIO=1` + `export TF_DS_ENGRAM_AIO=1` →
systemd restart → **`docker inspect` 实读 `TF_DS_ENGRAM_AIO=1`** → 段 B → trap 恢复并 restart。
health 200 分别在 48s / 47s 达成。**绝不并行两引擎。**

### 2.1 decode：全部在噪声带内

| 档 | AIO=0 | AIO=1 | 差 |
|---|---:|---:|---:|
| qa (think on) | 60.018 | 59.703 | −0.52% |
| code (think on) | 73.853 | 74.215 | +0.49% |
| prose (think on) | 61.714 | 61.498 | −0.35% |
| rand (think on) | 101.626 | 100.656 | −0.95% |
| boiler (think on) | 73.227 | 73.017 | −0.29% |
| qa (think off) | 62.066 | 62.047 | −0.03% |
| code (think off) | 91.489 | 91.005 | −0.53% |
| prose (think off) | 58.204 | 57.825 | −0.65% |
| rand (think off) | 113.329 | 114.166 | +0.74% |
| boiler (think off) | 102.189 | 102.583 | +0.39% |

**最大绝对漂移 0.95%，10 个档全部 <1%，且方向不一致（5 负 5 正）** ⇒ **无效应**。
`accept_rate` 与 `tok_per_round` 在两臂**逐位相同**（如 code-on 恒 0.668 / 2.8764）
⇒ **drafter 行为完全未变**，这是比时间更硬的否证证据。

### 2.2 prefill：效应巨大且复现历史

| 臂 | 三次原始 | 中位 |
|---|---|---:|
| AIO=0 | 2074.05 / 2049.41 / 2074.59 | **2074.05** |
| AIO=1 | 1748.74 / 1771.82 / 1767.63 | **1767.63** |

**AIO=1 比 AIO=0 慢 14.77%**（等价于 AIO=0 相对 AIO=1 **+17.33%**），
与历史 +19.5% 同量级同方向 ⇒ **prefill 收益真实且仍在**。

## 3. 判决

**`ENGRAM_AIO=0` 保留。它是 prefill 纯收益、decode 零代价。**
「关掉 Async 损害了 decode」这一假设**被双向干净干预否证**。

**方法论增量**：一个开关「同管 decode+prompt」时，**端到端否决只覆盖了权重大的那一侧**。
本例 prefill 侧收益 17% 足以主导端到端结论，而 decode 侧即使有代价也会被淹没。
**此类开关必须两侧分别立项验收**，否则留下未验证的 trade-off。本次补验结果是「无代价」，
但**缺口本身是真实的流程问题**，建议入手册。

## 4. 生产恢复

trap 恢复后实读 `TF_DS_ENGRAM_AIO=0`，`local.sh` sha256 前 16 位 **179b622b2b841fd9** 与备份一致，
health 200（47s），`check-boot.sh` **RESULT: OK**。

## 5. 复现

```bash
bash runs/forge-20261126-engram-decode/ab.sh   # 自带 trap 恢复
```
存证：`raw/{a0,b1}-{on,off}.log`、`raw/{a0,b1}-pf-{1,2,3}.json`、`raw/local.sh.bak`。