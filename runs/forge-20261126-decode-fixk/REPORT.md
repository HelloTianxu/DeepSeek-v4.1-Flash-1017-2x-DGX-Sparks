# E-DECODE-03：confidence 卡口量化 —— 放开全部负向，自适应即最优点

> 怀疑：单路 depth 上限 15（live=1），实测 tok/round 仅 2.05–4.74 ⇒ **是不是 confidence 过保守把 decode 限住了？**
> **判决：否证。强制放开草稿数（FIX_K=4/8）在 10 个档上全部负向。confidence 自适应就是最优工作点。**

## 0. 依据与更正

`runs/forge-20261124-decode/REPORT.md:107-111` 原文：
> `MultiDecoder._depth: the most drafts a stream verifies in a round of **live** streams`
> N=4 时上限 = 16/4−1 = **3**，实测均 1.336 ⇒ **当前不是上限在卡，是 confidence 在卡**

**关键更正（本轮）**：分母是 **live streams（实际活跃流）**，不是配置的 `TF_DS_STREAMS=4`。
**单路请求时 live=1 ⇒ 上限 16/1−1 = 15**，不是 3。
这解释了本轮实测 `tok_per_round` 达 **4.7407 > 4** ——
此前在 E-DECODE-02 中按配置 STREAMS 推出的「上限 4」是错的，**已撤回**。

## 1. 干预语义

`TF_DS_FIX_K=K` —— patch 注释原文：
*"a measurement: every drafted stream verifies K drafts a round"*（`0016:1450` 称 *a cost measurement*）。
即**绕过 confidence 自适应，强制每轮验证 K 个草稿**。verify 仍会拒绝错误草稿，故不改变输出正确性。

## 2. 三臂实测（同场，每臂重启并核验容器变量，trap 恢复）

`TF_DS_FIX_K` 实读分别为 **[]（未设）/ [4] / [8]**，health 200 均 46s。

### 2.1 thinking ON

| 档 | unset | K=4 | K=8 | K=8 相对 unset |
|---|---:|---:|---:|---:|
| qa | **60.188** | 54.695 | 51.456 | **−14.51%** |
| code | **74.674** | 69.263 | 66.149 | **−11.42%** |
| prose | **61.971** | 57.285 | 53.053 | **−14.39%** |
| rand | **101.543** | 94.490 | 95.432 | **−6.02%** |
| boiler | **73.349** | 71.747 | 69.134 | **−5.75%** |

### 2.2 thinking OFF

| 档 | unset | K=4 | K=8 | K=8 相对 unset |
|---|---:|---:|---:|---:|
| qa | **62.385** | 61.303 | 59.437 | −4.73% |
| code | **91.290** | 83.509 | 82.244 | −9.91% |
| prose | **58.007** | 49.228 | 45.677 | **−21.26%** |
| rand | **114.614** | 110.234 | 108.159 | −5.63% |
| boiler | **102.567** | 89.334 | 101.849 | −0.70% |

**10 个档全部负向，无一例外。**

## 3. 机制：为什么放开反而更慢

以 qa (think on) 为例：

| 臂 | accept_rate | tok/round | tok/s |
|---|---:|---:|---:|
| unset | **0.5822** | 2.048 | **60.188** |
| K=4 | 0.3199 | 2.2655 | 54.695 |
| K=8 | **0.2604** | 2.2857 | 51.456 |

**`tok/round` 只从 2.048 微升到 2.2857（+11.6%），而 `accept_rate` 从 0.582 崩到 0.260（−55%）。**
⇒ 强制多 drafts 产生的绝大部分是**无效草稿**，verify 阶段的计算被浪费，
每轮成本上升超过多产出的 token ⇒ **净亏**。

**结论：confidence head 的自适应深度不是「保守的限制」，而是已经调准的最优点。**
「单路 tok/round 远低于上限 15」**不是受限，而是最优**——上限高只说明有空间，不说明该用满。

## 4. 判决与连带

- **`TF_DS_FIX_K` 保持 unset**，不入库、不上生产。
- **「confidence 过保守导致 decode 受限」假设否证。**
- 连带否证「为 4 路配置牺牲了单路深度」这一猜测：depth 按 **live** 算，单路本就已拿到上限 15。
- **`ROUND_ROWS` 16→32 对单路无意义**（上限已 15 且用不满），与 E-KIT-01「单流零收益」一致。

## 5. 生产恢复

trap 恢复后实读 **`TF_DS_FIX_K=[]`、`TF_DS_ENGRAM_AIO=0`**，
`local.sh` sha 前 16 位 **179b622b2b841fd9** 与备份一致，health 200（46s），
`check-boot.sh` **RESULT: OK**（avail 9 GiB）。

## 6. 复现

```bash
bash runs/forge-20261126-decode-fixk/scan.sh   # 自带 trap 恢复
```
存证：`raw/fixk/{k-unset,k4,k8}-{on,off}.log`、`raw/fixk/local.sh.bak`。