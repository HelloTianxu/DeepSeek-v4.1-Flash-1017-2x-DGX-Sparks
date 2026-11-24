# E-COMM-01 · AllGather 归因与通信屋顶标定

日期 2026-11-25 · 底座 `base-alias`（pin `fbbc9713`）· 生产 prefill 地板 2084.42 tok/s
原始数据 `raw/`（Spark `~/ptxwork/` 取回）· 工具 `tools/{ab-nccl.py,comm-roof.sh,comm-ab.sh,nccl-ag-bench.cu,ag-bench-run.sh}`

---

## 0. 判决表

| 候选 | 判决 | 依据 |
|---|---|---|
| `NCCL_PROTO=Simple` | **否决** | 主项本就是 Simple（空操作）；decode **−3.6%**（剥夺小消息 LL） |
| `NCCL_IB_QPS_PER_CONNECTION=2` | 否决 | 噪声内 |
| `NCCL_IB_QPS_PER_CONNECTION=4` | **否决（更慢）** | +20% vs 同场 baseline |
| `NCCL_MAX_NCHANNELS=32` | 否决 | 偏慢 |
| `NCCL_MAX_NCHANNELS=16` | 否决 | A/B/A/B/A/B 配对差 −0.50% / −4.73% / **+1.12%**，不同向 |
| `NCCL_BUFFSIZE=8M` | 否决 | 偏慢 |
| 分块 + 拷回正确布局 | **否决（更慢）** | 2987.7 vs 单发 2723.7 µs，**+9.7%** |
| 分块 + 消费端直读分块布局 | **保留** | 2265.7 µs，−16.8%，**位一致**；估 +1.73% |
| **gather 载荷减半（fp16/bf16）** | **最大杠杆，待精度闸** | 2.410×，估 **+6.28%** |

**一句话**：协议/env 这一族**全部关闭**，且我 E-PLAN-01 的物理账有两处错。真正的杠杆是**字节数**，不是协议。

---

## 1. 协议杠杆为什么闭 —— 运行时决策日志

`NCCL_PROTO set by environment to Simple` **只证明 env 到达进程，不证明协议被采纳**。
`NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING` 打出的 `Protocol | LL | LL128 | Simple |` 只是**候选表头**，不是选择结果。

真正的身份证据是 TUNING 的**逐次决策行**：

```
AllGather: 41943040 Bytes -> Algo RING proto SIMPLE channel{Lo..Hi}={0..63}
```

按 systemd 启动时刻分段统计整个 pilot 窗口：

| Algo/proto | 次数 |
|---|---:|
| RING **SIMPLE** | **1166** |
| RING LL | 4 |
| PAT SIMPLE | 4 |

**default 臂的 41943040 B 主项也是 `proto SIMPLE`。** 4 次 LL 全是小消息。
⇒ NCCL 的 tuner **本来就为大消息选 Simple、为小消息选 LL**，已是合理选择。强制全局 Simple 只能剥夺小消息的 LL。

pilot 同场读数（`raw/comm-pilot/summary.json`）：

| | prefill 8192 tok/s | ttft_s | decode tok/s |
|---|---:|---:|---:|
| default | 1779.003 | 4.432 | **57.783** |
| Simple | 1780.850 | 4.414 | **55.700** |
| Δ | **+0.10%（噪声）** | | **−3.60%** |

### 1.1 对 E-PLAN-01 的两处更正

**更正 A —— 协议身份。** E-PLAN-01 §2 写「协议 RING_LL，大消息用低延迟协议」。
**错。** 那是 trace 里的**内核符号名** `ncclDevKernel_AllGather_RING_LL`，不是运行时协议选择。
运行时决策日志明确为 SIMPLE。**内核符号名 ≠ 运行时协议**。

**更正 B —— 分母。** E-PLAN-01 用「双口 200 Gb/s 均 ACTIVE ⇒ 50 GB/s 聚合 ⇒ 利用率 28.4%」。
**分母取错了。** 实读每个 CX7 功能的 PCIe 协商：

```
/sys/class/infiniband/rocep1s0f1/device   32.0 GT/s PCIe   width 4
/sys/class/infiniband/roceP2p1s0f1/device 32.0 GT/s PCIe   width 4
```

两个**独立 PCI 域**（`0000:01:00.x` / `0002:01:00.x`），各 Gen5 ×4。
单向理论 \(32 \times 4 \times \frac{128}{130} / 8 = 15.75\) GB/s，两域共 **31.51 GB/s** —— 比 50 GB/s 线速紧得多。
且实测峰值只有 18.9 GB/s（见 §2），**线速和 PCIe 理论值都不是可达分母**。

### 1.2 顺带查明的三件事

- **`NCCL_NET_PLUGIN=spcx` 是死配置**：`NCCL WARN Spectrum-X (SPCX) NCCL plugin is not supported on device:rocep1s0f1, skipping` + `Failed to initialize NET plugin SPCX` ⇒ 回落 `Assigned NET plugin IB to comm`。功能无损，但配置具误导性。
- **GDR 已启用**：rank 日志 `Connected all rings, use ring PXN 0 GDR 1`。
  我一度怀疑 GDR 没开（channel 行无 `/GDRDMA` 后缀、`cuMemGdrSupport 0`）——**自我否证**：`cuMemGdrSupport` 只关乎 NCCL 2.30 的 symmetric-memory 特性，与标准 GDR 无关。
- **`e.two`（`prompt.zig:171`）是四节点 2D 拆分**，两节点不适用。若将来扩到 4 台，`t.quarters` 会变成相关项（且它正是 §3 需要的分块布局范式）。

---

## 2. 通信屋顶标定

E-PLAN-01 留了「NCCL 真实可达上界未标定」这个缺口。本节补上。

### 2.1 两级工具

**① 镜像自带 `tf-dsv41-comm`**（`comm_main.zig`）：hidden 5120 fp32，计时 1/6/24/**2048 行**，
2048×5120×4 = **41,943,040 B 正是生产主项尺寸**（与 §1 决策行字节数精确相等，互为交叉校验）。
显存约 126 MB ⇒ **可与生产并存**，且 `docker exec` 传 env ⇒ **通信 A/B 从 4 min/臂降到 25 s/臂，无需重启生产**。

首次标定：**2492.2 / 2493.4 µs**（两 rank），`gather_ok=true`。

**② 自建 `tools/nccl-ag-bench.cu`**：直调 `ncclAllGather`，扫 8 个行数 × {fp32, bf16} + 分块臂，
CUDA event 包住 reps 次调用，逐臂小整数图案正确性闸（≤127，fp32/bf16 均精确可表示）。

### 2.2 噪声带（先界定再判决）

`tf-dsv41-comm` baseline 五次独立测量：**2492.2 / 2660.0 / 2727.2 / 2820.6 / 2634.3 µs**，跨度 **13.2%**。
⇒ 噪声带约 ±6%。§0 表中除 qps4（+20%）外全部落在带内，故一律判否决而非「未测出」。

### 2.3 带宽曲线（`raw/ag-r{0,1}.json`，22 臂全 `ok=true`，两 rank 一致到 0.3%）

| 字节 | fp32 µs | fp32 GB/s | bf16 µs | bf16 GB/s |
|---:|---:|---:|---:|---:|
| 0.492 MB | 92.5 | 5.32 | 42.2 | 5.83 |
| 2.621 MB | 240.2 | 10.91 | 105.9 | 12.37 |
| 10.486 MB | 604.3 | 17.35 | 304.9 | 17.20 |
| **20.972 MB** | **1150.8** | **18.22** | 565.6 | 18.54 |
| 26.214 MB | 1594.5 | 16.44 | 693.8 | **18.89** |
| 31.457 MB | 2084.4 | 15.09 | 838.9 | 18.75 |
| 36.700 MB | 2289.6 | 16.03 | 985.0 | 18.63 |
| **41.943 MB（生产）** | **2723.7** | **15.40** | 1130.4 | 18.55 |

**峰值可达 ≈ 18.5–18.9 GB/s**（= PCIe 两域理论 31.51 的 60.0%）。
**生产尺寸 41.94 MB 恰好落在掉档区（15.4 GB/s）。**

### 2.4 判决性对照：时间只由字节数决定

| 字节 | fp32 | bf16 | 差 |
|---:|---:|---:|---:|
| 10.486 MB | 512 行 **604.3** µs | 1024 行 **565.6** µs | 6.4%※ |
| 20.972 MB | 1024 行 **1150.8** µs | 2048 行 **1130.4** µs | **1.8%** |

※ 10.486 MB 那格 fp32 本轮偏慢；前一轮同配置为 566.4 vs 566.8（**0.07%**），见 `raw/ag-final.log` 与前一版读数。

同字节、不同 dtype、不同元素数 ⇒ 时间相同。**dtype 本身不影响传输时间，只有字节数影响。**
这条对照有两个直接后果：
1. 「减半字节」的收益是**实测的**，不是按 dtype 推的；
2. 该收益**可无损迁移到 fp16** —— fp16 与 bf16 同为 2 字节，时间必然相同，而 fp16 尾数 10 位 vs bf16 的 8 位。

### 2.5 生产距屋顶多远

生产主项 2.9550 ms/次 vs 孤立同尺寸 2.7237 ms ⇒ **生产已达孤立可达的 92.2%**，只有 **1.085×** 余量。
⇒ **「28.4% 利用率」是误导性分母造成的错觉；调度侧几乎没有空间。**

---

## 3. 分块实验：悬崖真实，但拷回一律更慢

带宽在 ~21 MB 以上掉档（§2.3）。假说：把 41.94 MB 拆成 2×20.97 MB 可回到 18.5 GB/s 档。

`2048 行 fp32` 拆成 S 个子 gather，两种落地方式各测（`raw/ag-r0.json` 的 `split*`）：

| S | pure（只发子 gather） | vs 单发 2723.7 | copy（落临时缓冲再拷进 `[2,2048,D]`） | vs 单发 |
|---:|---:|---:|---:|---:|
| 2 | **2265.7** | **−16.8%** | **2987.7** | **+9.7%** |
| 4 | 2370.4 | −13.0% | 2935.7 | +7.8% |
| 8 | 2607.3 | −4.3% | 3125.3 | +14.7% |

`copy` 臂全部 `ok=true`（逐元素校验图案），`pure` 臂标 `ok=null`（不落最终布局，仅计时）。

**假说成立但落地被否决**：拷贝成本 = 2987.7 − 2265.7 = **722.0 µs**，搬 83.9 MB（读+写 **167.8 MB**）
⇒ **232.4 GB/s = DRAM 屋顶 246.2 的 94.4%**。**拷贝已在 DRAM 屋顶，无压缩余地。**

⇒ **分块只有在消费端直接读分块布局（零额外拷贝）时才划算。**
这正是上游 `t.quarters` 的范式（`prompt.zig:1149`，四节点路径），我方两节点走的是 `e.comm.allGather` 连续布局。
改布局是**位一致**变换（同一批字节换个摆法），**零精度风险**。

---

## 4. 投向表（全局 Amdahl）

底：窗口 4972.654 ms、prefill 2084.42 tok/s、AllGather 主项 502.35 ms（**10.10%**）= 170 次 × 2.9550 ms。

| 杠杆 | 新主项 ms | 省 ms | prefill tok/s | Δ | 性质 |
|---|---:|---:|---:|---:|---|
| env / 协议 | 502.35 | 0 | 2084.42 | 0 | **已闭（实测）** |
| 完美调度（消掉生产 vs 孤立 8.5% 差） | 463.03 | 39.32 | 2101.03 | +0.80% | 上界，非直接可做 |
| **split2 + 消费端改布局** | 417.88 | 84.47 | **2120.44** | **+1.73%** | **位一致**，需改布局 |
| split2 + 拷回 | 551.04 | −48.69 | 2064.21 | −0.97% | **否决（实测更慢）** |
| **载荷减半 fp16** | 208.49 | 293.86 | **2215.34** | **+6.28%** | **需精度闸** |

**口径声明**：后三行是把**实测的孤立比值**单步施加到生产主项上的**上界估算**，
假设「生产随孤立 bench 同比缩放」。**不是实测的端到端收益，也不是比例连乘。**
落地必须重新同场 A/B 实测。

### 4.1 为什么 fp16 优于 bf16

gather 搬的是 **f32 累加 partial**（`_gemm` OUT 本身 fp32，`prompt.zig:1148` 注释
「[world, n, D] fp32 of every rank's partial, in rank order」）。消费端把两 rank 的 partial 相加。

- bf16 尾数 8 位 ⇒ 相对误差 ~\(2^{-9}\) ≈ 0.2%
- fp16 尾数 10 位 ⇒ ~\(2^{-11}\) ≈ 0.05%，**好 4 倍**，且字节数相同 ⇒ **时间收益完全相同**（§2.4 已证 dtype 不影响时间）
- fp16 风险是**范围**（±65504）而非精度，须先测 partial 的实际动态范围

**这推翻了我原先「gather bf16 再本地升 f32 位一致」的设想** —— 降 dtype 不可能位一致，必须过精度闸；
但**候选应是 fp16 而不是 bf16**。

### 4.2 与 E-PLAN-01 优先级的差异

E-PLAN-01 把「AllGather 协议 A/B」列为第 1 优先（理由：零代码、28.4% 利用率、从未测）。
**该理由的两个支撑点都被本轮推翻**（协议本已最优、分母取错）。
修正后的通信侧优先级：

1. **载荷减半 fp16**（+6.28% 上界，需精度闸 + 动态范围实测）
2. **split2 + 消费端布局**（+1.73% 上界，位一致，纯工程）
3. ~~env/协议~~ 关闭

E-PLAN-01 的第 2 项（**275 ms 归因**，短 prompt 下占 30%）与第 3 项（稠密链路 `TF_EXL3_PREFILL_TILES`）**未动**，
其中 275 ms 的绝对量级（275 ms vs 本轮最大杠杆 293.86 ms）**与 fp16 杠杆相当**，且**位置仍未定**。

---

## 5. 边界与自纠

**未做 / 不成立**
- 生产主项 2.9550 ms 来自 **2026-11-23 的 trace**，当前 `base-alias` 配置未重测内核时长；§4 全部估算继承该口径。
- 分块 `pure` 臂**不落最终布局**，是上界；真实落地还需消费端读分块布局的改造，其自身代价**未记账**。
- fp16/bf16 的**精度影响完全未测**，§4 末行只是带宽上界。partial 动态范围未测。
- 全部测量为**空载单流**。生产多路（head 最小空闲 4.8 GiB）下未复验。
- 带宽掉档（>21 MB → 15.4 GB/s）**机制未归因**，只观察到现象并绕开。
- `NCCL_ALGO` 未单独扫（2 rank 下 Ring 已是最优，Tree 无意义；PAT 仅 4 次出现）。

**本轮自纠（跑前/跑中拦下）**
1. `socket(AF_INET, SOCK_STREAM, 1)` —— 第三参是**协议号**，1=ICMP 与 SOCK_STREAM 不兼容 ⇒ socket 返回 −1、`bind: Bad file descriptor`。改 0。
2. **正确性闸抓到自己的 bug**：f32 臂 rows≥128 全部 `ok=false`。原因是 host 缓冲 `h` 被上一轮 gather 下载结果覆盖，下一轮又把「gather 结果」当本 rank 图案上传。bf16 臂每次重建图案故通过。**分离下载缓冲 `hd` 后 22 臂全过。**
   ⇒ 若只看时间就报数，会引用一次未过闸的运行。**§7.5 纪律在此生效。**
3. 「完美调度」行初版误用相对缩放（`iso/iso=1`）导致 saved=0；应取**绝对**孤立时间 170×2.7237。
4. 拷贝带宽初版只按单向 83.9 MB 算得 116.2 GB/s（47.2% 屋顶）；D2D 拷贝应按**读+写 167.8 MB** 计 ⇒ 232.4 GB/s（94.4%）。**结论方向相反**：拷贝不是「还有余地」而是「已到屋顶」。
5. GDR 假说自我否证（§1.2）。
6. worker 无 `~/ptxwork` 目录导致 scp 失败，脚本补 `mkdir -p`。

---

## 6. 复现

```bash
# 孤立屋顶（镜像自带，无需编译，约 25 s，可与生产并存）
scp tools/comm-roof.sh ztxjzh@192.168.1.118:/home/ztxjzh/ptxwork/
ssh ztxjzh@192.168.1.118 'bash /home/ztxjzh/ptxwork/comm-roof.sh'

# env 族扫描（7 臂，约 3 min，不重启生产）
scp tools/comm-ab.sh ztxjzh@192.168.1.118:/home/ztxjzh/ptxwork/
ssh ztxjzh@192.168.1.118 'bash /home/ztxjzh/ptxwork/comm-ab.sh'

# 尺寸 x dtype x 分块（22 臂，含正确性闸）
scp tools/nccl-ag-bench.cu ztxjzh@192.168.1.118:/home/ztxjzh/ptxwork/
ssh ztxjzh@192.168.1.118 'docker cp /home/ztxjzh/ptxwork/nccl-ag-bench.cu dsv41-1017-exl3-tf:/tmp/
  docker exec dsv41-1017-exl3-tf nvcc -O2 -std=c++17 -o /tmp/nccl-ag-bench /tmp/nccl-ag-bench.cu -lnccl
  docker cp dsv41-1017-exl3-tf:/tmp/nccl-ag-bench /home/ztxjzh/ptxwork/nccl-ag-bench'
scp tools/ag-bench-run.sh ztxjzh@192.168.1.118:/home/ztxjzh/ptxwork/
ssh ztxjzh@192.168.1.118 'bash /home/ztxjzh/ptxwork/ag-bench-run.sh'

# 协议身份（决定性证据，不是 env 自报行）
ssh ztxjzh@192.168.1.118 'journalctl --user -u dsv41-tensorfold.service --since "-30min" --no-pager \
  | grep -oE "AllGather: [0-9]+ Bytes -> Algo [A-Z]+ proto [A-Z0-9]+" | sort | uniq -c | sort -rn'

# 端到端协议 A/B（会重启生产，自动恢复）
ssh ztxjzh@192.168.1.118 'python3 /home/ztxjzh/ptxwork/ab-nccl.py --arms default,Simple \
  --prefill-sizes 8192 --suite prefill,decode --debug --outdir /home/ztxjzh/ptxwork/comm-pilot'
```

生产状态收尾：`health=200`、avail **10523 MiB**、`scripts/local.sh` 无哨兵残留。

---

## 7. 原始读数索引

| 文件 | 内容 |
|---|---|
| `raw/comm-pilot/summary.json` | 端到端 2 臂 pilot（prefill/decode/ttft） |
| `raw/comm-pilot/nccl-0{1,2}-*.txt` | 两臂 env 自报行 |
| `raw/comm-pilot/local.sh.pre-comm-ab` | 原始 local.sh 备份 |
| `raw/comm-roof-r{0,1}.json` | 首次屋顶标定 2492.2/2493.4 µs |
| `raw/comm-ab.txt`、`comm-ab-*.raw` | env 族 7 臂 |
| `raw/ag-r{0,1}.json`、`ag-final.log` | 22 臂尺寸×dtype×分块（全过闸） |
| `raw/nccl-ag-bench.cu` | bench 源码（Spark 副本） |
---

## 8. 恢复核验（2026-11-25，回答「生产是否已回到最佳基线」）

### 8.1 配置与运行态身份：**逐字节一致**

| 项 | 结果 |
|---|---|
| `scripts/local.sh` sha256 | `179b622b2b841fd9dca57adb4acef67234c071ea690fb7a8e847efffae51ba3d` = `scripts/local.sh.authoritative` **完全相同** |
| 镜像 | `tensorfold-dsv41:base-alias` ✓ |
| 服务 | `active` ✓ |
| 容器 NCCL env | 只剩标准五项（`IB_GID_INDEX/IB_HCA/NET_PLUGIN/SOCKET_IFNAME/VERSION`），**无 PROTO/ALGO/QPS/BUFFSIZE/DEBUG 残留** ✓ |
| 残留 bench 进程 | 0 ✓ |
| arena 自报 | **`arena 2208 MiB used`** ✓（`TF_DS_PROMPT_ALIAS` ON） |
| avail | 10488 MiB ✓ |

### 8.2 性能：**三项均在漂移带内**

**必须先对齐口径与请求形状**（见 §8.3）。

| 指标 | 基线 2026-11-24 | 本次（两轮） | Δ |
|---|---:|---:|---:|
| prefill 中位（10752 tok 冷前缀） | **2084.42** | **2081.41**〔2075.76 / 2083.33 / 2081.41〕 | **−0.14%** |
| N=1 decode | **59.265** | **59.676** / **60.008** | +0.69% / +1.25% |
| N=4 聚合 | **107.621** | **107.667** / **107.241** | +0.04% / −0.35% |
| arena | 2208 MiB | 2208 MiB | 0 |

MTP 正常：N=1 `tok_per_round 2.048`、`accept_rate 0.5848`；N=4 `accept_rate 0.680–0.684`。
文档记载本机同日漂移可达 1%，**三项全部落在带内 ⇒ 生产已回到最佳基线**。

### 8.3 **本轮最重要的一条自纠：基线表未标注口径与形状，导致我先得出错误的「回退」结论**

我先用 `tools/bench.py --suite prefill,decode,multi`（默认 2048/8192/32768）测，得到
prefill 中位 1663.90、decode 56.32、multi.4 118.61，与基线表 2084.42 / 59.265 / 107.621
分别差 **−20% / −5% / +10%**，看上去像严重回退。**全部是口径错配，不是回退：**

1. **prefill 2084.42 的真实出处**是 `runs/forge-20261124-alias-audit/raw/pf-al-on-2.json`：
   `prompt_chars 28038`、`prompt_tokens 10752`、`prefill_seconds_delta 5.158274`
   ⇒ 10752 / 5.158274 = 2084.418。它是**单一形状 3 次的中位**，
   走 **`cold_prompt.py` + `/metrics` 计数器增量**，**不是** bench.py 跨尺寸中位。
   该 JSON 的字段名还带 `server_tok_s_under_ncu_NOT_A_SPEED_CLAIM` 警示（本次无 ncu，
   `client_wall 5.175 ≈ prefill_seconds 5.158` 可证未发生 kernel replay）。
2. **decode 59.265 / 107.621 出自 `tools/probes/decode_probe.py` 与 `decode_conc.py`**，
   与 bench.py 的 `decode`/`multi` 套件**prompt 与计时口径都不同**（bench.py decode prompt 仅 37 token）。
3. 按同形状同工具重测后，三项立刻回到 ±1.3% 内。

**教训（已写入手册 §7.15 配套纪律）**：
**基线表必须同时标注「请求形状 + 测量工具 + 分母字段」**，否则跨工具对比会凭空造出 20% 的假回退。
本轮 `docs/CURRENT-BASELINE.md` 的基线表已补口径注。

### 8.4 核验原始数据

`raw/rv-pf-{1,2,3}.json`（cold_prompt 28038 目标字符，实际 28076）、
`raw/rv-dec-n1.jsonl`、`raw/rv-dec-n4.jsonl`、
`raw/restore-verify.{json,log}`（bench.py 那轮，**保留作为口径错配的证据，不作为基线判据**）。
