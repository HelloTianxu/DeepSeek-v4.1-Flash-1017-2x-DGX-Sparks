# E-DECODE-02：单路 decode「100 → 60」判决 —— 不是回退，是两重口径差异

> **⛔ 精确化批注（2026-11-26 同日晚，用户纠偏后）**：本报告原将 **thinking 开关**列为
> 「100→60」的主因之一，**该表述不准确，就此更正**：
> **thinking ON 下 `rand` 档实测 101.63 tok/s** ⇒ **thinking 开着照样能 100+**，
> 故 thinking **不是**「100 掉到 60」的原因。
> §3 的 code 档 +22.76% 仍然成立，但它只是**该内容档内**的效应，**不可外推为全局主因**。
> **唯一主因是生成内容的可预测性（MTP accept_rate）**：accept 0.58 ⇒ 2.05 tok/round ⇒ ~60；
> accept 0.86 ⇒ 4.34 tok/round ⇒ ~101。
> 另已补测并否证两条怀疑线，见
> [`forge-20261126-engram-decode`](<../forge-20261126-engram-decode/REPORT.md>)（ENGRAM_AIO 对 decode 无影响）与
> [`forge-20261126-decode-fixk`](<../forge-20261126-decode-fixk/REPORT.md>)（放开 confidence 全部负向）。
> **§6 关于 118.1 的「未复现」已查明来源**：`docs/measurements.md` 由上游作者 **MiaAI-Lab 于 2026-10-29**
> 引入（commit `0e7d421`），**118.1 / 80.7 是上游数值，不是我方实测**，我方项目 11-23 才开始。

> 用户报：**「之前我单路代码是可以跑到 100 tokens 每秒的，现在看起来只有 60」**。
> 结论：**引擎没有回退。100+ 现在就能跑到。** 60 与 100 之间隔着两个口径变量：
> **thinking 开关**（同场交替 A/B 实测 **+22.76%**）与**生成内容可预测性**（qa 62 → boiler 103，**1.66×**）。

## 0. 三句话结论

1. **不是回退**：生产二进制、镜像 `base-alias`、`ROUND_ROWS=16 / PARALLEL=4` 自 `61d51db` 引入后**从未改动**（git -L 核实）；同一时刻同一引擎，单路跨档实测 **56.90 – 113.91 tok/s**。
2. **100+ 可达且已复现**：thinking OFF 下 **boiler（样板代码）102.99**、**rand 113.91**、**code 90.92 tok/s**。
3. **权威基线 59.265 不是引擎单路上限**：它是 **thinking ON + B-tree 技术问答**这一特定口径的读数。**此前文档未标注 thinking 状态，是本次误读的直接来源。**

## 1. 机制：tok/s = rounds/s × tok_per_round，后者由 MTP 接受率决定

`decode_tok_s` 的差异**全部**落在 `tok_per_round` 上，而它由 `accept_rate` 决定 —— 即 **drafter 能猜中多少**。
`rounds/s` 不是常数：draft 更多会让每轮更慢，但净吞吐仍升。

| 档（thinking ON, 256 tok） | tok/s | accept | tok/round | rounds/s |
|---|---:|---:|---:|---:|
| qa | 59.75 | 0.5848 | 2.048 | 29.17 |
| code | 74.37 | 0.6680 | 2.876 | 25.86 |
| prose | 61.69 | 0.5948 | 2.170 | 28.43 |
| rand | 100.65 | 0.8559 | 4.339 | 23.20 |

**注意 rounds/s 从 29.17 降到 23.20** ⇒ 不能用「rounds/s 恒定 × tok/round」做线性外推。

## 2. 决定性发现：thinking ON 时被测的根本不是「生成代码」

抓完整响应结构发现 `message` 含 **`reasoning_content`**，`usage.completion_tokens_details.reasoning_tokens > 0`。
同场 A/B 中 **thinking ON 臂 `rtok=256`，恰等于 `max_tokens`** ⇒ **256 个 token 全部是思考独白，正式输出一个都没有**。

**所以「thinking ON 的 code 档」测的是模型内部推理独白的 decode 速度，不是代码生成速度。**
`tools/probes/decode_probe.py`（权威基线探针）**未传 `chat_template_kwargs`** ⇒ 服务端默认 thinking ON。
而 `tools/bench.py:115` 用 `"chat_template_kwargs": {"enable_thinking": c.thinking}`，默认 **off**。
**两套探针的 thinking 状态不同，数值本来就不该直接比。**

## 3. A 级证据：thinking 开关同场 A/B/A/B 交替（code prompt 固定）

`tools/probes/decode_think_ab.py`，同一 prompt、同一 max_tokens=256、温度 0，交替三对：

| 配对 | ON tok/s | OFF tok/s | 差 |
|---|---:|---:|---:|
| pair1 | 74.276 | 91.292 | **+17.016** |
| pair2 | 74.256 | 90.753 | **+16.497** |
| pair3 | 74.072 | 91.159 | **+17.086** |

**三对全部同向**，中位差 **+17.016**，ON 中位 **74.256** → OFF 中位 **91.159**，**+22.76%**。
机制量同步移动：`accept` 0.6680 → 0.7714、`tok/round` 2.8764 → 3.8788、`rtok` 256 → 0。
**干预生效已验到机制量本身**（手册 §7.20），不是只看总时间。

## 4. 内容分型全表（thinking OFF, 256 tok, 各 3 次取 median_excl_first）

| 档 | tok/s | accept | tok/round | prompt_tok | reasoning_tok | content_len |
|---|---:|---:|---:|---:|---:|---:|
| qa（B-tree 问答） | 61.96 | 0.5591 | 2.246 | 21 | 0 | 1165 |
| **code（红黑树）** | **90.92** | 0.7714 | 3.879 | 36 | 0 | 1022 |
| prose（散文） | 58.03 | 0.5694 | 1.882 | 29 | 0 | 1143 |
| **rand（续写随机词）** | **113.91** | 0.9349 | 4.741 | 60 | 0 | 1166 |
| **boiler（20 个 dataclass）** | **102.99** | 0.8138 | 4.741 | 72 | 0 | 985 |

**content_len 全部 ~1000 且 rtok 全 0** ⇒ 确认这一批真的在生成正式内容，干预干净。

### 4.1 thinking 开关效应依内容而定（跨批次，B/C 级，不可当 A 级用）

| 档 | ON | OFF | 变化 |
|---|---:|---:|---:|
| qa | 59.75 | 61.96 | +3.71% |
| code | 74.37 | 90.92 | **+22.24%** |
| prose | 61.69 | 58.03 | **−5.92%** |
| rand | 100.65 | 113.91 | +13.18% |

**prose 关 thinking 反而略慢** ⇒ 「关 thinking 一定更快」是错的。
**只有 code 档做了同场交替（§3，A 级）**；其余三档是不同批次对比，**仅作方向参考，不得外推**。

## 5. 长假设被否证（⛔ 已限定：仅 thinking ON 成立）

> **⛔ 更正（E-DECODE-04）**：本节结论**只在 thinking ON 下成立** —— 那时 token 全被思考独白吃掉，
> 延长只是延长独白。**thinking OFF 下延长假设成立**：4096 token 时 code **99.05（+8.26%）**、
> rand **127.64（+12.63%）**，accept 随模式稳定上升（rand 达 0.9959）。
> 见 [`forge-20261126-decode-upstream-align`](<../forge-20261126-decode-upstream-align/REPORT.md>)。

原猜「256 token 太短、没进入稳定可预测区，延长会爬向 118」。实测 **1024 token 反而略降**：

| 档 | 256 tok | 1024 tok |
|---|---:|---:|
| qa | 59.75 | 56.90 |
| code | 74.37 | 75.41 |
| prose | 61.69 | 58.90 |
| rand | 100.65 | 93.50 |
| boiler | — | 80.55 |

（此表为 thinking ON。）KV 增长带来的 attention 开销抵消了模式稳定收益。**延长生成不是提升单路 tok/s 的杠杆。**

## 6. 与历史文档的对账（诚实标注未复现项）

- `docs/measurements.md:62` 记我方单流 **code 118.1 / prose 80.7**。
  **本次同口径未复现**：code **90.92**、prose **58.03**。
  **未定位到 118.1 的原始测量脚本与形状**（prompt、max_tokens、KV 长度、thinking 状态均未知），
  故**不能判定文档有误，只能记为「当前形状下未复现，原始口径待定位」**。
- `docs/measurements.md:44-54` 说 bench.py 随机词 prompt「drafter 几乎无从预测」故只有 55-61。
  **本次 rand 档 accept 最高（0.9349），但这不能反驳该文档** ——
  我的 rand prompt 是「**续写同样模式的随机词列表**」，输出高度重复、极易预测；
  bench.py 是**用随机词构成问题**，生成内容不可预测。**两者不是同一种 prompt。**
- `E-DECODE-01` 记 N=1 **61.4**、本次 qa-on **59.75**、权威基线 **59.265** ⇒ **同口径一致，无回退**。

## 7. 已撤回的本轮臆断

- **「MTP depth = ROUND_ROWS/PARALLEL − 1 = 3，故 tok/round 上限 4」** —— 实测 `tok/round` 达 **4.7407 > 4**，推算无依据，**撤回**。引擎只自报 `drafts: on`，未自报 depth 数值。
- **「延长生成会让 accept 爬升」** —— §5 已否证。
- **「rand 档可代表 bench.py 随机词口径」** —— §6 已澄清，两者 prompt 设计不同。

## 8. 口径修正（已落地）

**权威 decode 数值必须同时标注 thinking 状态与 prompt 类型**，否则会被误读为引擎上限。
`docs/CURRENT-BASELINE.md` 的 decode 段已加口径限定块。

**单路 decode 的正确表述**：
- **thinking ON + 技术问答**：**59.265**（权威基线口径，用于回归对比）
- **thinking OFF + 代码生成**：**90.92**（code）～ **102.99**（boiler）
- **本次实测单路上界**：**113.91**（rand，高 accept 内容）

## 9. 复现

```bash
# Spark head
python3 tools/probes/decode_kind.py 256 3 off    # 五档 thinking OFF
python3 tools/probes/decode_kind.py 256 3 on     # 五档 thinking ON
python3 tools/probes/decode_think_ab.py          # code 档同场 A/B/A/B 交替
```

原始存证：`raw/decode-kind.log`（ON 256）、`raw/decode-kind-1024.log`（ON 1024）、
`raw/decode-kind-nothink.log`（OFF 256）、`raw/think-ab.log`（交替 A/B）。

## 10. 方法论增量

**新坑：推理模型的「生成速度」必须先确认 thinking 状态与 reasoning_tokens 占比。**
`max_tokens` 可能被思考独白全额吃掉（本次 ON 臂 rtok=256=max_tokens），
此时测到的是**推理独白的 decode 速度**，与用户感知的「代码生成速度」是两个量。
**判据：报 decode tok/s 时必须同时报 `reasoning_tokens` 与 `content_len`**，
否则不同探针间数值不可比 —— 建议入手册。