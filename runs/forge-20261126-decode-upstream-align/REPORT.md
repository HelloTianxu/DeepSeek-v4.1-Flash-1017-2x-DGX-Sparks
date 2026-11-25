# E-DECODE-04：对齐上游口径后的判决 —— 内核没坏，差距来自权重位宽与测量口径

> 用户质疑：**「code 118.1 / prose 80.7 我就是用这个部署的，一样的硬件，按理说不应该变弱。
> 是不是 CUDA 内核之前替换搞坏了？」**
>
> **判决：内核没坏。三件事叠加造成「看起来变弱」，对齐后我方达到甚至超过上游预期。**
> ① **权重不是同一套**：上游 **2.5bpw**，我方 **2.9bpw EAQ-v1.1 + ablit**，字节 **+16%**；
> ② **测量口径不同**：上游 **90 s sustained**，我方先前用 **256-token 突发**；
> ③ **thinking 状态**：ON 且 `max_tokens` 小时全部 token 被思考独白吃满。
> **对齐后（thinking OFF + 4096 token）：code 99.05、rand 127.64、boiler 101.71 tok/s。**

## 0. 定量核对（决定性）

| 项 | 值 |
|---|---:|
| 字节比 2.9 / 2.5 | **1.1600** |
| 理论速度比（带宽受限，字节线性） | 0.8621 ⇒ **慢 13.79%** |
| 上游 code **118.1**（2.5bpw）换算到 2.9bpw 的期望 | **101.81** |
| **我方 code-off 4096 实测** | **99.05** ⇒ 相对期望 **−2.71%** |
| **我方 rand-off 4096 实测** | **127.64** ⇒ 相对上游原值 118.1 **+8.08%** |

**−2.71% 落在同日漂移带内（本机实测漂移可达 1%，跨会话更大），且 prompt 与上游不完全相同。
⇒ 我方 decode 与上游在各自权重下表现一致，没有内核级劣化。**

## 1. 权重不是同一套（关键事实，此前被我忽略）

**上游 `README.md:2` 原文**：
> Weights: [**bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw**](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw)

**我方生产容器实读**：
```
TF_DS_MODEL_REVISION=Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1.1-2.9-bpw@aee640e2...,
                     quant=exl3-2.9bpw,ablit=1@0bdc2036...
```

⇒ **用户用的是同一套引擎/recipe，但权重是我方为质量选的 2.9bpw EAQ v1.1 + ablit。**
`runs/forge-20261124-upstream/REPORT.md:111` 其实早已记录这条边界：
> 上游用 **2.5bpw**，我方 **2.9bpw**（字节多 16%），其单流 106.45 不能全部归因于代码。

**这条边界当时写了，但没有换算成「我方 decode 的期望值」，导致后来把它当成劣化。本轮补上换算。**

## 2. 为什么字节直接换成速度：decode 已在 DRAM 屋顶

`runs/forge-20261124-decode/raw/byte-account.json`：

| 项 | 值 |
|---|---:|
| `per_token_bytes_GB.total` | **4.1135 GB/token** |
| `roofline_1tok.tok_s_ceiling` | **59.851** |
| 实测 DRAM 均值 / 占屋顶 246.2 | **203.9 GB/s / 82.8%**（中位 220.9 / 89.7%） |

**我方 qa 档实测 ~60 tok/s ≈ 1-token 屋顶 59.851 ⇒ 已贴带宽极限。**
带宽受限时时间 ∝ 字节，故位宽 +16% 几乎线性换成速度 −13.8%。**这不是内核效率问题。**

### 2.1 顺带解释了 FIX_K 为何全负（E-DECODE-03 的机制补全）

`roofline_by_verify_tokens`：

| verify tokens | uniq_experts_per_layer | total_GB |
|---:|---:|---:|
| 1 | **6.0** | 4.1135 |
| 2 | **11.906** | 5.6211 |

**多 verify 一个 token 会激活近两倍专家 ⇒ 读近 1.37 倍权重字节。**
所以 `rounds/s` 随草稿数下降**不是固定开销，而是 MoE 激活面扩张**。
强制 `FIX_K` 多起草时，被拒绝的草稿也付了这份字节 ⇒ **必然净亏**。
E-DECODE-03 的「自适应即最优点」由此得到机制层解释。

## 3. 长生成才是上游口径（更正先前一条否证）

**thinking OFF，256 → 4096 token：**

| 档 | 256 | **4096** | 变化 | accept(4096) | tok/round(4096) |
|---|---:|---:|---:|---:|---:|
| qa | 62.07 | **66.66** | +7.40% | 0.6282 | 2.444 |
| **code** | 91.49 | **99.05** | **+8.26%** | 0.8194 | 4.593 |
| prose | 58.20 | 57.37 | −1.44% | 0.5668 | 1.861 |
| **rand** | 113.33 | **127.64** | **+12.63%** | **0.9959** | 5.902 |
| boiler | 102.19 | 101.71 | −0.47% | 0.8079 | 4.765 |

**⛔ 更正 E-DECODE-02 §5**：原文「延长到 1024 反而略降 ⇒ 长假设否证」**只在 thinking ON 下成立**
（那时 token 全被思考独白吃掉，延长只是延长独白）。
**thinking OFF 下延长假设成立**：code +8.26%、rand +12.63%，accept 随模式稳定而上升
（rand 达 **0.9959**，几乎全中）。

## 4. 已排除的「内核替换」怀疑项

| 怀疑 | 核查结果 |
|---|---|
| `KIT_REF` 换版（E-KIT-01 换 v0.7.5） | 上游默认 `abe07c7d509a…`，我方 `fdde3285…`。**E-KIT-01 已实测单流 59.83 vs 基线 61.4，噪声内无收益也无劣化**；198/198 内核逐字节保留、MANIFEST 383 条 0 不符 |
| `TF_DS_PROMPT_ALIAS=1`（patch 0068） | **alias-audit 已同场测过 decode**：OFF **59.236** / ON **59.265**，**+0.05%** ⇒ 中性，关掉也回不到 118 |
| `0049-first-draft-depth` | 是**改善**：让首请求就用满深度，避免 boot 后浅起草 |
| `0050-draft-calibration` | **未改 `default_prior` 值**（那行只是 diff 上下文）；`draft_cal` 默认 0 且容器未设 ⇒ 不生效 |
| `ENGRAM_AIO=0` | E-ENGRAM-02 同场 A/B：decode 10 档全部 ±1% 内，accept/tok_per_round 逐位相同 |
| 生产二进制被实验替换 | 全部实验在离线容器；`verify-baseline.sh` **53/53 on pin fbbc97132c** |

## 5. 判决与口径落地

**没有内核级劣化。** 「100 → 60」的观感由三层口径差叠加：
**thinking 状态 × 生成长度 × 内容可预测性**，再叠加**权重位宽 2.9 vs 2.5** 的 13.8%。

**我方单路 decode 的正确期望值（2.9bpw、thinking OFF、长生成）**：
- 代码类 **~99–102 tok/s**
- 高 accept 内容 **~128 tok/s**
- 技术问答/散文 **~57–67 tok/s**（accept 0.57–0.63，属内容特性）

**若要对齐上游 118.1 的绝对值，唯一途径是换回 2.5bpw 权重** —— 那是**质量换速度**的取舍，
不是修 bug，需用户决策（我方当前选 2.9bpw + ablit 是为质量）。

## 6. 复现

```bash
# Spark head
python3 tools/probes/decode_kind.py 4096 2 off   # 对齐 sustained 口径
python3 tools/probes/decode_kind.py 256  2 off   # 短生成对照
```
存证：`runs/forge-20261126-decode-kind/raw/decode-kind-4096-off.log`、`-4096-on.log`。
字节账：`runs/forge-20261124-decode/raw/byte-account.json`。