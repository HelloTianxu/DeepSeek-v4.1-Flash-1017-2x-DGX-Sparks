# 更正：CX7 多宿主拓扑 —— 不是「单条 x4 卡死 200G」

**本文件更正 E-COLL-01 REPORT §3 与结论 §6.7b 的错误归因。**
用户指出后核实，原始测量数值全部正确，**错的是机制解释与「非配置可解」这个断言**。

## 1. 真实拓扑（本机实测 + NVIDIA 官方文档双向确认）

GB10 SoC 无法给网卡提供原生 x8，于是引出**两条独立 PCIe Gen5 x4 Root Port**
喂同一块 ConnectX-7，网卡内部聚合。**同一个物理口经两条 x4 各呈现一次**（"twins"）。

本机证据：

| 接口 | PCI | phys_port_name | phys_switch_id | carrier | speed |
|---|---|---|---|---|---|
| enp1s0f0np0 | 0000:01:00.0 | **p0** | dbba3c00034c1a44 | 0 | -1 |
| enp1s0f1np1 | 0000:01:00.1 | **p1** | dbba3c00034c1a44 | 1 | 200000 |
| enP2p1s0f0np0 | 0002:01:00.0 | **p0** | dbba3c00034c1a44 | 0 | -1 |
| enP2p1s0f1np1 | 0002:01:00.1 | **p1** | dbba3c00034c1a44 | 1 | 200000 |

**四个接口 `phys_switch_id` 完全相同；两个 ACTIVE 接口的 `phys_port_name` 都是 `p1`。**
⇒ `rocep1s0f1` 与 `roceP2p1s0f1` **是同一个物理口的两个 twin**，不是两个独立物理口。
两条 x4 分属 PCI domain 0000 与 0002（各自上游 `0000:00:00.0` / `0002:00:00.0`）。
**本机只插一根线（p1），p0 两口皆 DOWN。**

官方描述（spark-vllm-docker `docs/NETWORKING.md`）：
> "each physical port shares the same pair of PCIe5 x4 connections.
> Each PCIe 5 x4 link is represented by two Ethernet and two RoCE interfaces…
> Each of the twins represents one PCIe x4 link and can provide up to 100G link speed.
> in order to get full bandwidth in NCCL RDMA mode, we need to utilize **both** RoCE twins…
> `export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`"
> "connecting two Sparks using **both** ports won't give you any noticeable advantage"

NVIDIA 论坛官方回复亦确认：单口即可达 200G，两口共享同一 2×x4 池（理论 256 Gbps）。

## 2. 我错在哪

| 我的原话 | 实际 |
|---|---|
| 「200G 光口被主机侧 PCIe Gen5 x4 卡在 111.87 Gb/s」 | **单 twin** 走一条 x4 ⇒ ~112 Gb/s；**双 twin** 聚合 ⇒ ~185–200 Gb/s |
| 「这是拓扑事实，不是配置可解项」 | **是配置可解的**：只绑一个 PF 才锁死 126 Gbps，绑双 twin 即可突破 |
| 把 111.87 当作系统上限 | 111.87 只是**单 twin 的文档预期值**（NVIDIA 参考基准 111.71 Gb/s） |

**数值没错，机制错了。** 我把"单 twin 基准"误当成"整机天花板"，
而自己的双 rail 数据 185.19 > 126 本已构成反证，当时没顺着追。

## 3. 但本机并未踩坑：双 twin 已启用

容器实测环境：
```
NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1     ← 官方推荐的双 twin 绑定
NCCL_IB_GID_INDEX=5                      ← RoCEv2 + IPv4
TF_RDMA_DEVICES=rocep1s0f1,roceP2p1s0f1
NCCL_NET_PLUGIN=spcx
```
生产计数器佐证：`rocep1s0f1` xmit **5.8800 GB** vs `roceP2p1s0f1` xmit **5.8788 GB**，
**两条 twin 流量几乎完全均分** ⇒ 双 x4 确实在并行工作。

## 4. 更正后的天花板与结论

| 层级 | 理论 | 实测 | 达成 |
|---|---:|---:|---:|
| 单条 Gen5 x4 | 126 Gbps | **111.87 Gb/s** | 88.8% |
| 2×x4 池 | 252 Gbps | — | — |
| **物理口 p1 线速** | **200 Gbps** | **185.19 Gb/s（双 twin）** | **92.6%** |

**⇒ 真正的绑定约束是 200G 物理口线速，不是单条 x4。
我们已跑到其 92.6%，与社区 180–190 Gbps 的报告一致。**

**E-COLL-01 的判决因此不变、且根据更正确**：
生产可达 23.15 GB/s 是**双 twin 聚合实测值**，collective 已贴近硬件上限。
剩余理论空间 185.19→200 Gbps 仅 **+8% 网络**，折算端到端 **≤1%**，
且官方明确"实践中永远达不到"。**不值得追。**

MTU 结论亦不变：NVIDIA 自己的参考基准就跑在 **Mtu 1024**，得 111.71 Gb/s；
本机 1024→4096 实测带宽完全相同，端到端 −0.20%。

## 5. 对多机扩展的真实含义（用户指出，文档确认）

- **两机直连（当前）**：单缆 + 双 twin，已是最优，够用。
- **两口同时打满 200G**：做不到，共享 2×x4 池。
- **三机 daisy-chain**：每对节点只能维持 ~100G（官方原文）。
- **四机及以上**：需 QSFP 交换机，且聚合带宽池仍是瓶颈。

## 6. 来源

- NVIDIA 论坛：https://forums.developer.nvidia.com/t/clarification-on-optimal-use-of-interconnect-ports/370418
- 官方最佳实践：https://github.com/eugr/spark-vllm-docker/blob/main/docs/NETWORKING.md
- 本机 sysfs / ibv_devinfo / 容器 env 实测（本目录 `bw/`、`raw/`）
