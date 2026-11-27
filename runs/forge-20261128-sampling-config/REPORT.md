# E-SAMPLING-01 · 生产采样配置查明 + 实验采样参数硬约束

日期 2026-11-28 · 类型 **配置查明 + 纪律立项**（用户指令：不测，只记录）
证据 `raw/community-T0-vs-T1.webp`（外部视频截图，非我方实测）

---

## 0. 结论速览

| 项 | 值 |
|---|---|
| **生产服务端默认** | **temperature 1.0 · top_p 0.95 · top_k 20 · min_p 0（未设）** |
| 覆盖优先级 | **请求字段 > 服务端默认**；`temperature <= 0` 直接走 greedy，此时 top_p/top_k 不参与 |
| **新增硬约束** | **一切性能/质量测试与实验，必须用官方推荐采样 T=1.0 / top_p=0.95 / top_k=20，禁止 T=0** |
| **既有基线的口径缺口** | **现有全部性能基线数值均在 T=0（greedy）下测得**，与生产默认路径不同 |

---

## 1. 生产采样配置：四处证据一致

### 1.1 当前进程真实命令行（最硬证据）

```
tensorfold-native serve /hf/tensorfold-dsv41/aee640e2-ablit-0bdc2036/model \
  --backend cuda --host 0.0.0.0 --port 8888 --name DeepSeek-V4.1-Flash-1017 \
  --context 524288 --parallel 4 --max-tokens 32768 \
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  --thinking --api-key-file /run/secrets/tensorfold-api-key \
  --spill-dir /spill --spill-gib 64 --spill-min-free-gib 50 --spill-min-tokens 8192
```

取法：`docker exec dsv41-1017-exl3-tf bash -c 'tr "\0" " " < /proc/1/cmdline'`

### 1.2 引擎启动日志自报

```
[tensorfold] serving DeepSeek-V4.1-Flash-1017 at http://0.0.0.0:8888/v1
  (sampling: temperature 1.0, top_p 0.95, top_k 20; drafts: on; context: 524288; loaded in 32.0s)
```

出自 `serve.zig:198` 的 `shownSampling()`；该函数在 `sampling == null` 时打印 `greedy`。
**这一行是判断服务端默认的最快入口**，不必读命令行。

### 1.3 配置来源链（承重，改动入口在此）

```
scripts/config.sh:147-151
  # The family's measured sampling (BertholomusAI's two-node launch): temperature 1.0,
  # top_p 0.95, from the 20 likeliest tokens (the family's exact sampler needs a top-k).
  # A request's own values win.
  TEMPERATURE="${TEMPERATURE:-1.0}"
  TOP_P="${TOP_P:-0.95}"
  TOP_K="${TOP_K:-20}"
        ↓
start.sh:114   --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K"
        ↓
engine serve   config.default_sampling
```

- 三者均为 `${VAR:-默认}` 形式 ⇒ **`scripts/local.sh` 可覆盖**。
- **我方 `local.sh.authoritative` 未覆盖任何一项** ⇒ 生产用的就是这组官方默认。
- checkpoint 目录**没有 `generation_config.json`**（已核实不存在），
  故 `server/main.zig:167 sampling()` 的「读 generation_config」那一路**无贡献**，全部来自 CLI flags。

### 1.4 请求覆盖行为（实测三种请求）

| 请求 | 返回 `tensorfold.sampling` |
|---|---|
| 不带 temperature | **`exact`**（用服务端默认 1.0 / 0.95 / 20） |
| `temperature: 0` | **`greedy`** |
| `temperature: 0.7, top_p: 0.9` | **`exact`**（请求值优先） |

`exact` = 该 family 的精确采样器（`core/lanes/sampling.zig` 注释：与 Python `exact_sampling.py` 位对位）。

### 1.5 三层「默认」必须分清（易错点）

| 层 | temperature | top_p | top_k | 说明 |
|---|---:|---:|---:|---|
| `core/lanes/sampling.zig` 结构体默认 | 1.0 | 0.95 | 20 | **引擎内部结构体，服务端不直接用它的缺省** |
| `server.zig:resolveSampling` 代码缺省 | **0** | **1.0** | 20 | **仅当 default_sampling 与请求都没给时**；temp≤0 ⇒ 返回 null ⇒ greedy |
| **我方生产实际（CLI 注入）** | **1.0** | **0.95** | **20** | ← **这一层才是生产生效值** |

> **只读 `resolveSampling` 的 `orelse 0` / `orelse 1.0` 会得出错误结论**（我第一次读源码即误判为「默认 greedy / top_p 1.0」）。
> 必须回到**进程命令行或启动日志自报行**确认。

---

## 2. 新增硬约束：实验与测试一律用官方推荐采样

**用户指令（2026-11-28，原话）**：
> 「约束无论测试还是实验运行，也要使用官方推荐参数」

### 2.1 约束正文

**一切性能测量、A/B 实验、质量评测，必须使用 `temperature=1.0, top_p=0.95, top_k=20`（官方推荐 / 生产默认）。
禁止用 `temperature=0` 作为性能或质量结论的依据。**

理由：生产对外服务的默认路径就是 `exact` 采样。用 greedy 测出来的数，
**既不代表生产的吞吐，也不代表生产的输出质量**。

### 2.2 唯一例外：位一致性校验

`tools/exact.py`、`tools/validate.py` 以及内核探针的位一致闸**保持 greedy**。
它们的目的是**确定性比对**，不是性能或质量结论。
**这类运行不得产出任何 tok/s 结论**，报告中须标注「位一致校验，非速度口径」。

### 2.3 受影响工具清单（下次实验前必须先改，本轮按指令未改）

**A 类 — 写死 `temperature: 0`，无命令行开关，必须改代码：**

| 文件:行 | 用途 |
|---|---|
| `tools/bench.py:113` | 主力性能 bench（prefill/decode/multi/needle） |
| `tools/probes/cold_prompt.py:87` | **基线 prefill 口径工具**（10752 token 冷前缀） |
| `tools/probes/decode_kind.py:77` | decode 分类探针 |
| `tools/probes/decode_think_ab.py:34` | thinking A/B |

**B 类 — 有 `TEMP` 参数但默认 0.0，可命令行传值，建议改默认：**

| 文件 | 调用形式 |
|---|---|
| `tools/probes/decode_probe.py` | `decode_probe.py <label> [max_tokens] [temp] [reps]` ⇒ 第 3 参传 `1.0` |
| `tools/probes/decode_conc.py` | `decode_conc.py <N> [max_tokens] [temp] [reps]` ⇒ 第 3 参传 `1.0` |
| `tools/probes/decode_rep.py` | 同上 |

> **B 类只传 temp 还不够**：`top_p` / `top_k` 未随请求发送，会落到服务端默认（0.95 / 20），
> 恰好等于官方推荐值，故结果正确；但**报告须写明实际生效的三个值**，不能只写 temp。

**C 类 — 保持 greedy（见 §2.2）**：`tools/exact.py:137`、`tools/validate.py:49`。

### 2.4 既有基线的口径缺口（**重要，未解决**）

`docs/CURRENT-BASELINE.md` §4 的全部权威数值，以及 E-SELF / E-COMM / E-PLAN 系列的**所有性能读数**，
**均在 `temperature=0`（greedy）下测得**：

- prefill 中位 **2084.42**（`cold_prompt.py`，写死 T=0）
- N=1 decode **59.265**、N=4 聚合 **107.621**（`decode_probe.py` / `decode_conc.py`，默认 temp 0.0）
- E-COMM-01 恢复核验的 2081.41 / 59.676 / 107.667 等，同样 T=0

**这些数值仍是「同口径可相减」的有效历史基线**（彼此之间口径一致），
但**不能声称代表生产默认采样下的性能**。

**待办（用户已明示本轮不测）**：在官方采样下重建一次基线，并与 greedy 基线并列记账。
在此之前，任何引用旧基线的收益百分比，**必须标注「greedy 口径」**。

---

## 3. 外部证据：T=0 在低 bpw 量化思考模型上会烧光思考链

**来源**：B 站 UP 主「景星华光」视频截图，标题《调参反转①：DeepSeek 2.8bpw 写代码，千万别用 T=0!》
**画面自报环境**：`LIVE BENCH 2× NVIDIA DGX Spark · 256GB · TensorFold TP=2` —— **与我方硬件与并行方式同构**。
**证据性质**：**外部第三方实测，非我方实测**。原文截图存 `raw/community-T0-vs-T1.webp`。

| 任务 | T=0.0（贪婪） | T=1.0 / top_p=0.95（官方采样） |
|---|---|---|
| T3 手写递归下降解析器（12 单测） | **0 / 12 FAIL · 160.4 s** | **12 / 12 PASS · 126.8 s** |
| T4 并发限流器修 Bug（6 单测） | **0 / 6 FAIL · 203.2 s** | **6 / 6 PASS · 55.3 s** |

- 失败机制（画面原文）：**「2.8bpw 量化思考模型在贪婪解码下极易触发思考链复读循环」**，
  两题均 **16,384 思考 token 全烧光**，一题未输出代码、一题交白卷。
- 画面另称 T4 比 GLM-5.3（120.0 s）快 2.2 倍。
- 画面彩蛋：其评测脚本把右结合 `(1-5)^2^3` 即 `(-4)^8=65536` 误写成 4096，
  模型算出 `3.0001220703125` 反而暴露了测试集自身的 Bug。

### 3.1 引用边界（**不得越界**）

1. **bpw 不同**：画面是 **2.8bpw**，我方 checkpoint 是 **2.9bpw**（`…-EXL3-EAQ-v1.1-2.9-bpw`）。
   **机制方向可借鉴，具体数字不可搬为我方结论。**
2. **任务不同**：对方是代码单测通过率与端到端耗时，我方基线是 tok/s。**两者不可换算、不可相减。**
3. **未在我方复现**：本轮按用户指令未做任何测试。
   「我方 2.9bpw 在 T=0 下是否同样复读」**属未验证命题**，不得写成已知事实。
4. 该证据的作用是**支撑「实验必须用官方采样」这条约束的合理性**，
   以及提示**质量维度上 greedy 可能不只是「慢一点」，而是直接不可用**。

---

## 4. 本轮做了什么 / 没做什么

**做了**：查明生产采样配置与来源链（进程命令行 + 启动日志 + config.sh + 三种请求实测）；
保存外部证据；立此约束并写入三份权威文档。

**没做（用户明确指令）**：任何性能测试、任何 A/B、任何工具代码改动。
§2.3 的工具改造与 §2.4 的基线重建**均为待办**。

**自纠**：上一轮（回答「温度和 top-p 是多少」时）已查明配置却**未落文档**，
违反了用户长期要求「注意关键信息、流程、问题、方法、归因和思路的记录，以便后面来构造我们的一键部署脚本」。本轮补齐。
另：首次读 `resolveSampling` 源码时按 `orelse` 缺省误判为「默认 greedy、top_p 1.0」，
后由进程命令行与启动日志纠正（见 §1.5）。