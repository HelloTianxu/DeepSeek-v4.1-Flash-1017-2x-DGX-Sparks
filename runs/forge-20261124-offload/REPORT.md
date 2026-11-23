# E-OFFLOAD-01：权重卸载 SSD 换 chunk 8192 —— 可行性测算

> **判决：不成立。冷权重只有 1.46 GiB/rank，缺口 5.40 GiB（覆盖 27%）；
> 改用卸载热专家则在完美重叠假设下仍净亏 26.6%。**
> 但本轮查出一处真实浪费：`embed.weight` BF16 **两 rank 各存全量 1.233 GiB**。

用户提问：「SSD 速度/IOPS 空间还有很大，内存权重还有没有那部分用的不频繁、体量小
可以卸载到磁盘的？这样 8192 不就有空间了吗」

**全程只读**（sysfs / safetensors 头 / 源码 / dd 直读）。服务未停，内存 8.28–8.30 GiB 稳定。

---

## 0. 结论表

| 问题 | 实测/推导 | 判定 |
|---|---|---|
| 专家权重有"不频繁"子集吗 | chunk 2048 下期望冷专家 **3.8×10⁻¹²** 个 | **没有，100% 热** |
| 每 chunk 读多少专家字节 | **99.27 GB = 缓存的 97.2%** | 全量重读 |
| 真正的冷权重有多少 | **1.464 GiB/rank** | 覆盖缺口 **27.1%** |
| chunk 8192 结构缺口 | **10.20 GiB** 增量 − 4.8 可用 = **5.40 GiB** | 差 **3.7×** |
| 卸载热专家凑数 | SSD 1.019 s vs 原 core 0.805 s | **净亏 26.6%** |
| 引擎支持部分驻留吗 | `cuda.DeviceBuffer.alloc` + 全量 upload | **无分页路径** |

---

## 1. 核心前提被否：MoE prefill 的专家访问是稠密的

**这是整个思路的命门。** "不频繁"在 decode 成立（单 token 只路由 6/384），
**在 prefill 完全不成立**。

均匀路由下单专家在一个 chunk 内一次都不被选中的概率 \(p=(1-6/384)^{\text{cap}}\)：

| chunk | 期望活跃专家 | 单专家全冷概率 | 期望冷专家数 |
|---|---:|---:|---:|
| 2048 | 384.0 | 3.8×10⁻¹⁴ | **3.8×10⁻¹²** |
| 4096 | 384.0 | 9.7×10⁻²⁹ | 3.7×10⁻²⁶ |
| 8192 | 384.0 | 9.4×10⁻⁵⁷ | 3.6×10⁻⁵⁴ |

**既有实测独立佐证**（`layer-shape-ledger.json`，R=2048）：
`active_experts_uniform_hypothesis: 384.9999999999962`（384 routed + 1 shared）。

每 chunk 唯一专家字节 **99.27 GB**，占 95.1 GiB 缓存（102.11 GB）的 **97.2%**。
⇒ **每个 chunk 都把整个专家库重读一遍，没有跨 chunk 的时间局部性可利用。**

这正是"chunk 越大越快"的机理：**专家字节与 chunk 大小无关**，
加大 chunk 是把同一份权重读摊到更多 token 上。也因此，**卸载任何专家 = 每 chunk 都要从盘回读**。

---

## 2. 真正的冷权重清单（源码逐项确认）

`weights.zig` 加载点与切分方式（ref f77d5a1，**非生产 pin fbbc9713**，仅作结构参考）：

| 张量 | 尺寸 | dtype | 每 rank | 切分 | prefill 访问 |
|---|---|---|---:|---|---|
| `embed.weight` | [129280, 5120] | **BF16** | **1.233 GiB** | **否，replicated** | 仅行 gather 一次 |
| `head.trellis` | [320, 8080, 96] | I16 | 0.231 GiB | 是，vocab 半分 | 仅末 token 一次 |
| `vision.*` + `aligner.*` | — | — | **0** | — | **weights.zig 无加载点** |
| `mtp.*` | — | — | ~0.40 GiB | 部分 | **decode 投机需要，不可卸** |

**冷权重合计 1.464 GiB/rank。**

两个意外发现：
1. **vision tower 根本没加载**（`grep vision|aligner|visual weights.zig` 零命中）。
   这是个多模态 checkpoint（vision 0.77 GiB + aligner 0.14 GiB），但纯文本服务不加载它。
   **这条路没有可回收的余量——引擎已经做对了。**
2. **`embed.weight` 是 BF16 未量化且 replicated**：`L.plain("embed.weight", null)`，
   两个 rank 各存一份全量 1.233 GiB。而 `head` 已按 vocab 切分。**这是真实的不一致。**

---

## 3. chunk 8192 的真实缺口：5.40 GiB，不是 2.56

用 `chunk-mem/REPORT.md` §2.3 的结构上界法（arena 常量项仅 284 MiB，其余随 cap 线性）：

| chunk | arena | 相对 2048 增量 |
|---|---:|---:|
| 2048 | 3766 MiB | — |
| 4096 | 7248 MiB | +3.40 GiB |
| **8192** | **14212 MiB** | **+10.20 GiB** |

生产可用 = 4.8（用户基线）− 2.5（memguard 地板）= **2.3 GiB**；
CONTEXT 524288→262144 再释放 2.5 ⇒ **4.8 GiB**。

**缺口 = 10.20 − 4.8 = 5.40 GiB。冷权重 1.464 GiB 只覆盖 27.1%。**

即使 CONTEXT 再砍到 131072（产品能力 4×缩水）多释放约 1.25 GiB，
加上冷权重全卸仍差约 2.7 GiB。**够不到。**

---

## 4. 那卸载热专家凑数呢？——完美重叠下仍净亏 26.6%

实测盘能力（本轮，direct IO 只读）：

| 方式 | 带宽 |
|---|---:|
| 单流 dd 2 GiB | **4.0 GB/s** |
| 4 路并发 2 GiB | **5.69 GB/s** |

盘型号 YMTC PC450-2TB-D（NVMe，1.9T，42% 已用）。

**关键比值：专家核有效带宽 123.34 GB/s ÷ SSD 5.69 GB/s = 21.7×。**
每挪 1 字节到盘，取回时间贵 21.7 倍。

卸载 5.40 GiB（= 5.80 GB）后，每 chunk：

| 项 | 时间 |
|---|---:|
| DRAM 侧 (99.27−5.80)/123.34 | 0.758 s |
| **SSD 侧 5.80/5.69** | **1.019 s** |
| 完美重叠取 max | **1.019 s** |
| 原专家核 | 0.805 s |
| **变化** | **+26.6%（净亏）** |

**盈亏平衡点 = 0.805 s × 5.69 GB/s = 4.27 GiB。需要 5.40 GiB > 4.27 GiB。**
⇒ **即使假设完美重叠，也越过了平衡点。** SSD 占空比 100%，无任何抖动容忍。

**而这还是乐观上界**，两个已测事实使它更差：
1. 专家核只跑到 DRAM 屋顶的 **50.1%**（123.34/246.2），是**占用率/延迟受限**
   （REG 255、16.7% occupancy，见 Review §2）。**减字节不线性省时间**——
   这正是此前"减字节不省时间"的归因。所以 DRAM 侧不会真降到 0.758 s。
2. **本机已实测：往 prefill 路径加异步磁盘 IO（`ENGRAM_AIO=1`+`LEAD=1`）比 `AIO=0` 更差。**

---

## 5. 物理前提：引擎没有分页路径

`weights.zig` 的 `upload()`：

```zig
var b = try cuda.DeviceBuffer.alloc(L.d, bytes.len);
try b.upload(0, bytes);
```

**每个权重都是真实 CUDA 设备缓冲，启动时全量灌入。**
内核签名取 `u64` 设备指针（`trellis: u64, gate_ptr: u64`）。

`readWhole` / `span_buf` **只是加载期的 direct-IO 优化**（绕开 page cache、
避免乱序访问打掉 readahead），注释写明其页在池 sizing 前就还给系统：
> `span_buf: ?dio.Buffer = null, // page_allocator's, so its pages go back to the system before the pool is sized`

⇒ **运行期不存在"缺页从盘取"的机制。** 卸载需要新建：
staging 缓冲 + 预取调度器 + 改所有内核发射点。这是一个**新引擎子系统**，
不是配置项——参照 spill tier 用了 0088/0110/0111 三个 PR。

**另注**：既有 `SPILL_GIB=64` **不是权重卸载**，是 prompt 状态（KV）持久化，
避免重算 prefill（`PORT-RECIPE.md`：*"kept prompt states written to each Spark's own disk…
read back instead of a new prefill"*）。两者无关。

---

## 6. 本轮唯一有价值的副产品

**`embed.weight` BF16 replicated 1.233 GiB/rank 是不必要的。**
- `head` 已 vocab 切分，`embed` 却没有；
- prefill 只做行 gather，10752 token 仅需 129280 行中的 **8.3%**；
- 若像 head 一样 vocab 切分，**每 rank 省 0.617 GiB，且不需要磁盘**。

但：仍是引擎改动（需加 all-gather 或改 gather 路径），且 0.617 GiB
**不足以让 chunk 4096 免掉 CONTEXT 减半**（4096 需 3.40 GiB，2.3+0.617=2.92 仍不够）。
**列为观察项，不建议现在动。**

---

## 7. 边界与未做

- 源码 ref **f77d5a1 ≠ 生产 pin fbbc9713**，加载结构仅供定性；`upload` 全量驻留的结论
  与引擎自报 `read whole` 日志一致，但**未在生产 pin 上逐行复核**。
- arena 结构估计沿用 chunk-mem 的 ~92% 上界法，**真实值不确定性无法在构建前消除**。
- SSD 只测了顺序读 4.0/5.69 GB/s，**未测 4K 随机 IOPS**（若走 embed 行 gather 路线才需要）。
- 专家密度用均匀路由假设；真实路由更集中，只会让"全热"结论更强，不会反转。
- **未做任何写操作、未停服、未改配置。**

## 8. 产物

- `offload_account.py` / `offload-account.json` — 可复跑测算
- `probe_weights.py` — checkpoint 全量分类
- `probe_nonexpert.py` — 非专家张量逐项模式与尺寸