# E-DRAM-01 r2：2.00× 缺口已分离——不是 DRAM 重读，是内核在低占用下跑不满带宽

日期 2026-11-24，Asia/Shanghai。授权原话："先A后B吧，这个环境你是可以停机操作研究的 我可以暂时用云上模型"。

**r1 是事故，不是数据**：模型驻留态用 ncu 多 pass kernel replay 打爆 head 内存，
用户人工拔电源恢复。根因与硬约束见 [发现与结论 §0](<../../docs/performance/01-发现与结论.md>)、
[RESTORE.md §6](<RESTORE.md>)。**r1 未取得任何 DRAM 字节数据。**

**r2 改用零注入路径**：引擎由 systemd 照常运行，**不重启、不改配置、不注入引擎进程**；
观测用**设备级 GPU PM 采样**（`nsys --gpu-metrics-devices`），无内核重放。
采样容器带 `--memory=2g --memory-swap=2g` 硬上限。**全程 `free -g` available 稳定 8 GB，前后无变化。**

---

## 1. 方法与两个必须记住的口径陷阱

设备被 nsys 识别为 **Blackwell GB20B | NVIDIA GB10**，自动选中 metric set **GB20y**；
本轮用 `gb20y-top`（177 个计数器）。采样 5 kHz（200 µs），`--trace=none --sample=none`。

**陷阱一：GB10 没有 `dram__*` 计数器，且 `aperture_device` 恒为 0。**
统一内存机的 LPDDR 访问被归类为 **sysmem 而非 device vidmem**。
r1 实测 `lts__t_sectors_aperture_device_lookup_miss.sum` = **0**，
若直接采信就会把"零流量"当成事实。

**陷阱二：标着 `[GB/s]` 的字段，sqlite 原始值其实是 sectors/s。**
必须 `GB/s = raw × 32 / 1e9`。**用已知字节数的 streaming 内核校准两次**：

| 采样频率 | 已知期望 | 原始均值 | 换算后 | 误差 |
|---|---:|---:|---:|---:|
| 10 kHz | 245.73 GB/s | 7.6931e9 | **246.18 GB/s** | **0.184%** |
| 5 kHz | 243.25 GB/s | 7.6198e9 | **243.83 GB/s** | **0.241%** |

换算与频率无关（速率量），两次一致。证据 `calib-10k.json`、`calib-5k.json`。
校准中 L2 读命中率 1%、Total≈Misses，与纯流式自洽。

**计数器对**：`VidL2 Total from L1TEX` = SM 侧向 L2 请求的字节；
`VidL2 Misses from L1TEX` = 穿过 L2 去打内存的字节。两者之比即"重读是否被 L2 吸收"。
GB10 另有 SysL2（本级 chunk 内活跃度 30.4%），故 Misses 是**实际 DRAM 流量的上界**——
这个方向只会加强下面的结论。校准中 512 MB 工作集下 Misses 与已知 DRAM 流量吻合，
说明工作集远大于缓存时该口径成立；本轮专家权重每 chunk 49.6 GB，同样满足。

## 2. 请求与分段（实测）

冷前缀 prompt：nonce 置 0 位，`cached_tokens=0`，`max_tokens=1`。
**10755 未缓存输入 token / server prefill 6.376158 s = 1686.75 tok/s**（`cold-r2.json`）。
这是**零注入下的真实速度**，不是 r1 那种被 replay 污染的数字。

> 口径更正：`cold-r2.json` 里的键名 `server_tok_s_under_ncu_NOT_A_SPEED_CLAIM` 是 r1 遗留的**误名**，
> r2 全程无 ncu 注入，该值 1686.75 **是有效速度**。原始文件按"实验只增不改"保留，
> 脚本已改为 `server_tok_s` + `profiler_injected: false`。
> 与既有 1735 tok/s 基线的差别来自 token 数不同（10755 vs 8628，chunk 数 5 vs 4），**不是回归结论**。

繁忙期自然分成 **5 个约 877–942 ms 的区段 + 1 个 321.8 ms 尾段**，
正对应 10755 = 5×2048 + 515 的 chunk 结构。最大繁忙 chunk = 941.6 ms。

## 3. 判决：H1（tile 重读打到 DRAM、已在屋顶）被否证

E-ROOF-01 留下的两个竞争解释：
- **H1**：实际 DRAM 流量 ≈ 2× 唯一字节，专家核心**已在** DRAM 屋顶，可回收的是"消除重读"。
- **H2**：123 GB/s 就是真实流量，距屋顶还有一倍，属内核实现问题。

**H1 的定量预测**：每 chunk 专家核心 20 层 × 20.162 ms = **403.2 ms**，占 941.6 ms 的 **42.8%**；
若 H1 成立，这 42.8% 的时间里 DRAM 必须贴近 **246 GB/s**。

**实测（整个 chunk，不依赖任何相位门控）**：

| 量 | 实测 |
|---|---:|
| DRAM 均值 | **121.5 GB/s** = 屋顶 246.2 的 **49.3%** |
| DRAM 中位 / p90 / 最大 | 124.0 / 207.8 / 265.3 GB/s |
| 时间占比 > 150 GB/s | **18.7%** |
| 时间占比 > 200 GB/s | **约 9.8%**（200–220 档 4.3% + 220–240 档 5.6%） |
| 最长的高带宽连续段 | **3.6 ms**（不是 14 ms 或 8 ms） |

**没有任何约 400 ms、贴近 246 GB/s 的平台。** 直方图在 40–240 GB/s 之间宽分布，
众数落在 140–160 档（15.9%）。**H1 要求的形态不存在，H1 被否证。**

**⇒ H2 成立：E-ROOF-01 算出的 123.1 GB/s 基本就是真实 DRAM 带宽，
2.00× 是内核效率缺口，不是"已在屋顶、只能靠消除重读"。**

## 4. 重读确实存在，但被 L2 吸收了

按 **Tensor Pipe 活跃度**门控定位 MMA 相位（不用"带宽高"选样本，避免循环论证）：

| 门控 | 段数 | 总时长 | 占 chunk | DRAM 均值 | L2 请求均值 | L2 读命中率 | 实测 Occupancy | SM Issue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tensor>10% | 124 | 497.4 ms | 52.8% | 112.8 | 688.1 | 83.2% | 25.6% | 34.9% |
| **Tensor>20%** | 150 | **360.8 ms** | **38.3%** | **115.5** | **742.2** | **84.5%** | **24.2%（中位 17%）** | 36.2% |
| Tensor>40% | 30 | 37.4 ms | 4.0% | 117.1 | 1104.1 | 92.9% | 17.0% | 13.2% |
| Tensor>60% | 16 | 19.6 ms | 2.1% | 120.0 | 1169.5 | 93.7% | 16.9% | 13.0% |

（带宽单位 GB/s；Tensor>20% 的 38.3% 与既有时间账里专家核心占 **38.38%** 数值接近，
但**这不等于证明二者同一**——attention/HC 也用 MMA，且专家内核解码密集段 Tensor 活跃度低。
故 §3 的判决**不依赖**本表。）

**L2 请求是 DRAM 的 6.43×（742.2 / 115.5），读命中率 84.5%。**
所以"重读"是真的，但它**没有变成 DRAM 流量**——这正是 H1 推理链断掉的地方：
重读 ≠ DRAM 重读。

## 5. 机制候选：低占用下的延迟受限（不是带宽饱和，也不是发射饱和）

Tensor>20% 相位里三个饱和度**同时都不高**：

- DRAM **115.5 / 246.2 = 46.9%**
- SM Issue Stage **36.2%**（Tensor>60% 时只有 **13.0%**）
- Tensor Pipe Active **37.0%**

而**实测 Occupancy 中位数只有 17%**，与既有真实 trace 的静态驻留上界吻合：
down 内核 255 regs/thread、128 threads/block ⇒ **8/48 warps = 16.7%**；
gateup 106 regs、256 threads、44288 B shared ⇒ 16/48 = 33%。
**这是第一次用实测占用率证实静态推算，而不是停留在"寄存器压力是候选"。**

> **【2026-11-24 核实批注，E-SELF-02；原文保留】**
> 上面的静态账**已用二进制实测核实，成立**，但**内核身份需补明确**：
> 255 regs 那个 down 是 **kit 预编译 cubin 里的
> `tf_exl3x::grouped_rows_kernel<2,8,4,1,2,16,2,false>`**（及 `<…,2,10,2,false>`），
> 不是引擎源码里的任何 kernel。`cuobjdump -res-usage` 实测：
> **REG:255、STACK:0（零 spill）、SHARED:33920 B**，W=4 ⇒ 128 threads/block。
> 按 48 SM / 48 warps/SM / 65536 regs/SM：255×128=32640 ⇒ **2 blocks = 8/48 warps = 16.7%**，
> 与实测 achieved 17% 吻合 ⇒ **寄存器确实是 prefill 专家核心的限制因子，本节归因成立。**
> 需补的两点：
> ① 「gateup 106 regs、256 threads、44288 B shared」这行**家族串了**——
> 那个形状属于引擎侧 **Nemotron** 的 `tf_experts_prefill::prefill_kernel`（实测 98 regs），
> 我方 dsv41 不调用它；我方 gate/up 走 `tf_exl3x::grouped_cp_kernel`（decode，128 regs）
> 与 `grouped_mma_kernel`（prefill）。
> ② 该 cubin **无 .cu 源码**（kit 只发二进制），且引擎硬筛 `PF=1,G=2,FOLD=0`，
>  shipped 变体里**没有**同形状的低寄存器替代件 ⇒ **降寄存器这条自研路不可实施**。
> 证据与完整变体表见 `runs/forge-20261124-self-occupancy/REPORT.md`。

三条饱和度都低 + 占用率低 ⇒ **在飞的内存请求不够多，盖不住延迟**，
所以 DRAM 只跑到屋顶的一半。**这是延迟受限的特征，不是带宽受限。**

**仍未分离**：具体 stall 原因（long scoreboard / barrier / 解码依赖链）需要
stall-reason 计数器，PM 采样给不了；驻留态又禁止 kernel replay。
**本轮不宣称已唯一识别 stall 根因。**

## 6. 对 B（加大 chunk）的影响：机制上更站得住

原先支持加大 chunk 的理由只有"字节摊销"（每 token 专家成本 ≈ 权重读 ÷ M）。
现在多了一条**机制性**理由：既然内核是**低占用延迟受限**而非带宽饱和，
增大 M 让每次权重 tile 载入摊到更多 MMA 工作上，
提高算术强度并给内存管线更多重叠机会——**收益不必以"打满带宽"为前提**。

同时 §4 提示一个**新的候选次级瓶颈**：L2 请求已达 742–1170 GB/s。
若占用率或 M 提上去，限制可能从 DRAM 转到 L2。**本轮未测 L2 屋顶，不做预测。**

E-ROOF-01 的 chunk 条件投影（≈2271 tok/s）**仍然只是条件投影，未 A/B**。

## 7. 另一条本轮顺带观测到的线索（未追）

chunk 内 **GPC 频率均值约 2238.6 MHz**（原始值 2238558811，字段标 MHz 实为 Hz，
又一处单位名不实），而该卡上限 3003 MHz。**prefill 期间并未跑在最高时钟。**
GB10 的 CPU/GPU 共享功耗预算，这可能是另一条独立杠杆。
**本轮未做功耗/温度/降频归因，不当结论。**

## 8. 边界

- PM 采样是**整个 GPU**的吞吐，不是单内核字节数；§3 用形态否证 H1，不宣称精确的专家内核字节。
- `VidL2 Misses` 含读与写，且经 SysL2 后才是 DRAM，故它是 DRAM 流量**上界**。
- 只测 head（rank 0）；worker 未同期采样。
- 单请求单次，未做重复与噪声区间。
- 未测 L2 屋顶、未测 stall reason、未做 chunk A/B。
- 原始证据已从 `/tmp` 移到 Spark **`~/.cache/forge-dram-20261124/`**（`/tmp` 重启会清空）：
  `pm-prod-r2.nsys-rep`（9.17 MB）、`calib-r1.nsys-rep`、`calib-5k.nsys-rep` 及对应 JSON。
  派生结论已落盘本目录 JSON。