# 收尾核验：口径单一可追溯 + 生产最佳版本状态（2026-11-26）

> 对应用户诉求：**「先保证项目口径单一可追溯吧，提交后确保生产恢复最佳版本状态」**。
> 本轮**不做新实验、不改生产二进制**，只做审计、修正、核验、提交。

## 0. 三句话结论

1. **口径已单一可追溯**：新增 `tools/audit-consistency.py` 作为机器可执行判据，**AUDIT_PASS**（0 处未批注失效表述、10 个唯一口径关键值全覆盖）；指针链由「02 → 01 → CURRENT-BASELINE」收敂为**全部文档 → CURRENT-BASELINE 单一权威**。
2. **生产在最佳基线状态**：四道闸全过（重建性 53/53、开机自启 11/11、口径 AUDIT_PASS、arena 2208 MiB 自报），性能三项漂移 **+0.42% / +1.52% / −0.20%**，均在带内。
3. **生产从未被本轮实验触碰**：E-ROWS 等全部实验在离线容器内完成，生产镜像 `base-alias` 与 `start.sh` 未改动；离线容器已停，avail 10 GiB。

## 1. 口径审计：三层判定与真实修正

审计器 `tools/audit-consistency.py` 的判据分三层，避免把「复盘时引用旧结论」误判为口径污染：

| 层 | 判据 | 作用 |
|---|---|---|
| 文档级 | 头注含「历史快照 / 不是当前口径 / 原文保留」等豁免词 | 整篇定性为历史文档 |
| 段落级 | 命中行 ±40 行窗口内有 ⛔ / 撤回 / 否证 / 失效 批注 | 局部已更正即豁免 |
| 引用级 | 复盘、手册反例、否证台账中的引用 | 引用本身不是主张 |

初扫 **19 处命中**，三层判定后 **15 处为误报**（复盘/手册反例引用、或窗口内已有批注），**真需修正 4 处**：

| 坐标 | 失效内容 | 处理 |
|---|---|---|
| `01-发现与结论.md:75`（原） | N=8628 旧 tok 口径 | §2 节首加 ⛔ 块，列当前权威全套数值 |
| `02-航海日志.md:113` | N=8628 | 头注 ⛔ 声明整篇为历史快照 |
| `02-航海日志.md:115` | 旧天花板 2815.83 | 同上，并逐项列出失效清单 |
| `02-航海日志.md:398` | 旧天花板 3604.89 | 同上 |

**另修一处指针链缺陷（本轮新发现）**：`02-航海日志` 原写「当前有效口径以 **01-发现与结论** 为准」，
而 01 自身亦含历史数值 ⇒ **权威指向不单一**。已改为直接指向 `CURRENT-BASELINE.md`，
并在 01 头注声明「不再经本页中转」。

**唯一口径关键值覆盖（审计器强校验，CURRENT-BASELINE 必含全部）**：
tok **10762**、span **5157.783 ms**、折算 **2086.56 tok/s**、基线 **2084.42**、
专家核心 **2167.061 ms / 42.02%**、注意力HC **906.308 ms / 17.57%**、
4000 缺口 **300.222 ms**、arena **2208 MiB**。

## 2. 生产核验：四道闸

| 闸 | 脚本 | 执行位置 | 结果 |
|---|---|---|---|
| 重建性 | `tools/verify-baseline.sh` | **本地 Mac** | **53/53 patches applied cleanly on pin `fbbc97132c`** |
| 开机自启与运行态 | `tools/check-boot.sh` | **Spark head** | **11/11 全过，exit 0，RESULT: OK** |
| 口径单一 | `tools/audit-consistency.py` | 本地 Mac | **AUDIT_PASS** |
| 内存口径 | 引擎自报行 | Spark | **`arena 2208 MiB used`** |

`check-boot.sh` 11 项明细：local.sh == 权威副本 / Linger=yes / unit enabled / WantedBy=default.target /
docker enabled / **/health 200** / head image base-alias present / **TF_DS_PROMPT_ALIAS=1（patch 0068 active）** /
**TF_DS_L2_PREFETCH=1（worth +6.3% single stream）** / 容器内无被否决或实验开关 / **avail 10 GiB ≥ 6 GiB headroom**。

### 2.1 本轮踩坑并已固化：三脚本执行位置不同

**`verify-baseline.sh` 必须在本地 Mac 跑** —— 它需要上游 clone `../_upstream/tf`（本地存在，HEAD `ea64a870d`，含 pin）。
本轮先误在 Spark 上执行，得到 `FAIL: upstream clone not found`，**那是资产位置问题，不是基线损坏**。
Spark 侧无该 clone 且 `git fetch` GitHub 有 TLS GnuTLS -110，**不要在 Spark 上补 clone**。
反之 `check-boot.sh` / `deploy-baseline.sh` 必须在 Spark head 项目目录下跑。
已写入 `docs/CURRENT-BASELINE.md` §7 顶部警示块。

> 同时更正一条历史误记：此前曾记「本地 `_upstream/tf` 不存在」，**实测存在**（`_upstream/` 下有 `kit`/`prod-kit`/`tf` 三项，`tf` 与 `kit` 均为 git 仓库，`kit` HEAD `fdde328` 即 `KIT_REF`）。

## 3. 性能三项复测（同工具同形状，禁用 bench.py 跨尺寸中位）

| 指标 | 工具与形状 | 本次 | 权威基线 | 漂移 |
|---|---|---:|---:|---:|
| prefill 中位 | `cold_prompt.py <out> 28038` ×3，`/metrics` 增量 | **2093.25 tok/s** | 2084.42 | **+0.42%** |
| N=1 decode | `decode_probe.py final-n1 256 0.0 3`，`median_excl_first` | **60.163 tok/s** | 59.265 | **+1.52%** |
| N=4 聚合 | `decode_conc.py 4 256 0.0 3`，`agg_tok_s_median` | **107.401 tok/s** | 107.621 | **−0.20%** |

**三项均在 ±1.6% 漂移带内 ⇒ 在基线。**

- prefill 三次原始 **1906.44 / 2093.25 / 2098.28**；**首跑偏低是冷启动**，与历史两次核验同型（1932.21 首跑），按既有口径取中位。
- N=1 三次 57.877 / 60.109 / 60.216，`median_excl_first` = 60.163；MTP accept_rate 0.5796 / 0.5848 / 0.5848，tok_per_round 恒 2.048。
- N=4 三次 105.5 / 107.442 / 107.361，中位 107.401；accept_rate 0.676–0.684，per_stream 中位 26.85。
- 全程 `mem_after` 10.14–10.38 GiB，**NRestarts=0，无 OOM**。

## 4. 资源清理

| 容器 | 本轮前 | 本轮后 |
|---|---|---|
| `dsv41-1017-exl3-tf`（生产） | Up 3 hours | **Up，未动** |
| `exl3-rows`（E-ROWS 离线） | Up 28 minutes | **已 stop** |
| `exl3-sim`（E-SIM 离线） | Exited (137) | 保持 Exited |

avail **10 GiB**，与 `check-boot` 门槛一致。

> 生产内存硬约束提醒：head **多路最小空闲 4.8 GiB** 才是实验硬上限基线，
> 当前 10 GiB 是**单路空闲态**读数，**不可据此放宽离线实验的内存上限**。

## 5. 本轮未做的事（明确边界）

- **未改任何生产源码、patch、镜像、`start.sh`、`local.sh`**。
- **未新增实验结论**：性能主因口径仍是 E-ROWS-01/02 定案的**结构性配平、无单点主因**，本轮不动该判决。
- **未推进注意力/HC 归因**（17.57% / 906.308 ms / 1438 次调用，从未归因）——那是下一步战场，需用户确认转向。
- 非位一致授权、量化格式更换、扩容决策仍待用户表态。

## 6. 复现

```bash
# 本地 Mac
python3 tools/audit-consistency.py            # 期望 AUDIT_PASS
bash tools/verify-baseline.sh                 # 期望 53/53 on pin fbbc97132c

# Spark head（项目目录下）
bash tools/check-boot.sh                      # 期望 11/11 RESULT: OK
bash ~/ptxwork/final-verify.sh                # 三项复测，脚本副本见本目录 verify.sh
```

原始存证：`raw/pf-{1,2,3}.json`。