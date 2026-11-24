# E-SELF-03：prefill 专家内核延迟的根因 —— 供给侧是寄存器单级缓冲，不是带宽

**判决：根因已定位到代码。** 我方 prefill down 走的 `grouped_rows_kernel` 用
**`__ldg` 阻塞式 global→寄存器、缓冲深度 PF=1**；而**同一个文件里我方 decode 走的
`grouped_cp_kernel` 用 cp.async→shared 多级环形 + minBlocks=4**。
同一块硬件、同一个内存系统：**decode 打到 DRAM 屋顶 82.8%，prefill down 只有 49.3%。**
差别在供给侧设计，不在硬件。

> 立项由来：用户否掉「延迟受限」这个说法——**「时延是因为时延」是同义反复，不是归因**。
> 要求指出背压的具体成因：为什么生产跟不上消费。本报告回答这个问题。

---

## 0. 先接受两条用户纠正，并撤回我两个错误论断

### 0.1 时钟不是卡点，而且我用错了分母

用户主动降频并做过配置，**升频实测无收益**。我此前把「prefill 期间 2238.6 MHz / 上限 3003」
当成一条杠杆，**两处都错**：

1. **分母错**。NVIDIA GB10 白皮书 Table 4 明确 **GPU boost clock = 2.4 GHz**，
   不是 nvidia-smi 报的 3003 MHz。实测 2242 MHz ≈ **官方 boost 的 93.4%**，本就接近标称。
2. **方向错**。升频无收益这件事**本身就是延迟受限的正面证据**：
   stall 等的是内存延迟（固定墙钟），升频只让它占更多 cycle，墙钟不变。
   **用户已经做过的这个实验，比我任何推理都强，我却没把它当证据用。**

**⇒ 时钟杠杆永久关闭。**

### 0.2 L2「没测过」不等于「没有官方数据」

白皮书 Table 4：**Global (L2) cache = 24 MiB**，48 SM、6144 CUDA core（128/SM）、
**48 Tensor core（每 SM 仅 1 个）**、4 GPC。
官方**不给 L2 带宽**，所以「L2 屋顶未测」这句话对带宽成立、对容量不成立——我表述含混。
容量 24 MiB 是已知量，可直接用于判断驻留可行性。

---

## 1. 木桶不是一块短板，是每个子系统各有短板

完整时间账（`docs/performance/01-发现与结论.md` §2，实测）：

| 子系统 | 占窗口 | 短板归因状态 |
|---|---:|---|
| 专家 core | **38.38%** | ✅ **本报告已归因到代码**（§2–§4） |
| **no-kernel（GPU 无任何 kernel 在跑）** | **13.49%** | ❌ **完全未归因** |
| collective（含 peer 等待） | 10.84% | ❌ 未归因，需两 rank 同期采样 |
| other GEMM | 10.13% | ❌ 未归因 |
| HC | 8.01% | ❌ 未归因 |
| attention 名称内核 | 5.20% | ❌ 未归因（名称≠整个子系统） |
| rotation | 4.64% | ❌ 未归因 |
| 专家 epilogue/combine | 3.97% | ❌ 未归因 |
| unpack | 1.99% | ❌ 未归因 |
| other GPU | 3.69% | ❌ 未归因 |

**「没有任何单元饱和」不是「没有瓶颈」，而是「每个子系统各自被自己的依赖卡住」的表征。**
Tensor 相位三个饱和度同时低（DRAM 46.9%、SM Issue 36.2%、Tensor Pipe 37.0%）——
每个单元都在等，所以没有一个满。

**已算过的硬上界**：core 归零 + 全部 idle 归零仍只 3604.90 tok/s < 4000 目标。
**⇒ 4000 不可能只靠专家内核达到**，必须同时动 no-kernel 与 collective。

---

## 2. 根因：生产-消费在同一个 warp 里串行，缓冲只有一级

源码坐标：上游 `c99083245` 的 `src/tensorfold/cuda/exl3/experts_grouped.cuh`。
我方 prefill down = `grouped_rows_kernel`（:409）→ `warp_tiles_rows`（:219）。

### 2.1 供给侧是阻塞式 `__ldg`，不是 cp.async（代码实读）

```cpp
template <int K2>
__device__ void load_words(uint32_t (&dst)[Fmt<K2>::LW], const uint32_t* p, int lane) {
    for (int l = 0; l < Fmt<K2>::LW; ++l)
        dst[l] = __ldg(p + l * 32);      // ← global → 寄存器，无 shared、无 async
}
```

### 2.2 缓冲在寄存器里，深度 = PF，我方 PF=1（代码实读）

```cpp
uint32_t pf[PF][NT][LW];                 // ← 缓冲区就是寄存器数组
for (int ib = 0; ib < nkt; ib += PF) {
  for (int d = 0; d < PF; ++d) {
    w[i][l] = pf[d][i][l];               // 取本级
    if (it + PF < nkt)
      load_words(pf[d][i], tp + (it+PF)*kstride + ...);   // 只提前 PF 级
    for (int i = 0; i < NT; ++i) {
      decode_tile<CB,K2>(w[i], map, lane, b0, b1);        // 生产：反量化，纯 ALU
      for (int g = 0; g < G; ++g) {
        mma16816(acc[g][i][0], a[g], b0);                 // 消费：Tensor Core
        mma16816(acc[g][i][1], a[g], b1);
      }
    }
  }
}
```

**掩盖一次 global load 延迟的时间窗 = PF 个 tile 的 (decode + MMA)。PF=1 ⇒ 只有一个 tile。**
生产与消费**在同一个 warp 内按程序序串行**，没有 warp specialization。
跨 warp 的重叠只能靠 SM 上其它 warp——**只有 8 个**（2 blocks × 4 warps）。

### 2.3 为什么缓冲不能加深：寄存器被累加器占死（结构推算）

```cpp
float acc[G][NT][2][4];                  // G=2, NT=8 ⇒ 2×8×2×4 = 128 个 fp32 寄存器
```

`REG:255`（cubin 实测）里 **约 128 个是 MMA 累加器**，占一半。
其余是 `pf[1][8][LW]`（8–16）、`w[8][LW]`（8–16，**pf 的一份纯拷贝**）、
`a/an[G][4]`（16）、寻址与展开临时量。

> **口径**：128 这个数是**从数组维度结构推算**（全展开必须驻留寄存器），
> 不是 SASS 逐条统计。REG:255 / STACK:0 / SHARED:33920 是 cubin 实测。

**⇒ 闭环背压：**
累加器吃掉一半寄存器 → 缓冲只能 PF=1 → 在飞请求不足 → 盖不住 L2 延迟
→ Tensor Core 空转 63% → DRAM 只跑到 49.3%
→ **想加 warp 就得减累加器，而减累加器就是减复用** → E-SELF-02 路 A 实测亏 0.92% / 2.38%。

**这解释了路 A 为什么必然失败：在这个内核里「降寄存器」和「降复用」是同一件事，
因为寄存器主要被累加器占着。二者结构性耦合，不是可以独立调的两个旋钮。**

---

## 3. 决定性对照：同一个库的 decode 路径已经用了正确设计

| 我方路径 | 内核 | `__launch_bounds__` | 供给侧设计 | 实测 DRAM |
|---|---|---|---|---:|
| **decode** | `grouped_cp_kernel`(:1705) | **`(W*32, 4)`** | **cp.async → shared 环形，每 warp S 级**（:1651–1677） | **203.9 GB/s = 82.8% 屋顶** |
| **prefill down** | `grouped_rows_kernel`(:409) | `(W*32)` **无 minBlocks** | **`__ldg` → 寄存器，PF=1** | **121.5 GB/s = 49.3% 屋顶** |
| prefill gate-up | `grouped_mma_kernel`(:704) | `(256)` | cp.async → shared S 级（:847–861） | 未单独分离 |

**同一块 GB10、同一个 LPDDR5x、同一个作者。decode 82.8%，prefill down 49.3%。**
**⇒ 「显存速度不够」被这个对照直接排除。差别在供给侧的工程设计。**

本文件共 6 处 cp.async 惯用法（`cp_async16` / `commit_group` / `wait_group`，:572–578），
**基础设施现成，不需要从零写**。

---

## 4. 真正的杠杆：把缓冲搬出寄存器，不动复用

**关键：shared 有大量空闲，而且用它不花一个寄存器。**

我方 served 变体 shared 构成（代码实读 + cubin 实测）：

| 项 | 字节 |
|---|---:|
| `red[W][16][NT*16]` fp32 | 32768 |
| `rows_sh[16*G]` int32 | 128 |
| 编译器附加 | 1024 |
| **合计（cubin 报）** | **33920** |

2 blocks/SM 时每 block 可用 `102400/2 = 51200 B` ⇒ **空闲 17280 B**。

每级 cp.async 缓冲 = `W × NT × TW × 4 B`（`TW = 4·K2`）：

| K2 | TW | 每级字节 | 空闲区可放级数 |
|---:|---:|---:|---:|
| 6 | 24 | 3072 | **5** |
| 8 | 32 | 4096 | **4** |
| 10 | 40 | 5120 | **3** |

**⇒ 可以在完全不碰累加器、不碰复用、不碰总字节的前提下，把缓冲深度从 1 提到 3–5。**
这正是「预载 / 加大缓冲区」那条工程路线，且**与路 A 的失败机制正交**。

### 4.1 cubin 里没有现成件，必须自建

解码全部 24 个 `grouped_rows` 变体（`raw/` 表）：

- **PF 取值集合只有 {1, 2}**
- **PF=2 且 NT=8 且 G=2 且 FOLD=0 的变体：0 个**（PF=2 的 NT=8 件全是 G=1）
- 我方 served 两件 `<2,8,4,1,2,{10,16},2,false>` 都是 PF=1

**⇒ 加深缓冲这条路无法靠切 env 试，必须自建 cubin。**

### 4.2 上游有更新版 down 内核，但不是 drop-in

`grouped_down3_kernel`(:1417) 用 `__launch_bounds__(256, MINB)`，cubin 里有 3 件，
**REG:126–128、STACK:0、SHARED 仅 1296 B**——资源账漂亮得多。
**但签名不兼容**：输出 `__nv_bfloat16* Y`、输入 `const half* Xd`、且融合了 epilogue；
我方走 fp32 `Z` 且 epilogue 分离。`grouped_mma3_kernel` 同样是 bf16 路径。
**⇒ 不能直接换，需要适配层或改我方 epilogue，属更大改动。**

---

## 5. 下一步的优先级（按证据强度与代价）

1. **no-kernel 13.49% 归因** —— 纯观测、零改动、零内存风险，且是唯一「消掉即变吞吐」的时间。
   **完全未开垦，应最先做。**
2. **prefill down 供给侧改造（PF 加深 / cp.async 入 shared）** —— 根因已明、
   有同库 decode 的成功对照、shared 余量已量化。需自建 cubin，
   **前置闸：未改动自建 cubin 须与 kit cubin 位一致 IDENTICAL**。
3. **collective 10.84% 归因** —— 需两 rank 同期采样，判断是链路搬运还是负载不均等待。
4. ~~路 B：同形状加 minBlocks~~ —— **期望值进一步下调**：§2.3 已证寄存器被累加器占死，
   压上限只会 spill，不解决供给深度。**它治的是占用率，不是根因。**
5. ~~时钟~~ —— **永久关闭**（§0.1）。

---

## 6. 边界与未证事项

- §2.3 的「128 个寄存器是累加器」是**结构推算**，未做 SASS 逐条归属统计。
- §3 的 DRAM 对照来自**不同实验的不同相位**（decode 实验 vs prefill chunk 采样），
  **不是同场同请求的 A/B**；作为机制对照有力，作为定量差值需谨慎。
- 「PF 加深能拿到多少收益」**未测，不做数值预测**。禁止比例连乘。
- shared 余量计算基于 2 blocks/SM 的当前占用；若占用变化，余量随之变化。
- 本报告**未改任何代码、未做任何 A/B**，纯归因。生产基线不动。