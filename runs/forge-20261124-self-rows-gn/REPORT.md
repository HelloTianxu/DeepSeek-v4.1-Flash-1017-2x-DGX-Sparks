# E-SELF-04：位一致闸通过，路 B 编译期否决，G×NT 不变线开测

2026-11-24 至 25，Asia/Shanghai。承 E-SELF-03 §7 的逐层归因（根因在「造币工序」：权重解码的 ALU/洗牌没有被任何东西缓存，64 个程序逐位重复解同一份权重）。

本轮按用户指令「在两极上按你的思路行动，哪怕收益不正向也能基于结果再次识别根因」执行。**先跑便宜的一极，且把「跑」提前到编译期**——这是本轮最大的方法变化。

---

## 0. 结论速览

| 项 | 判定 | 硬判据 |
|---|---|---|
| 位一致闸（PTX 往返） | **通过** | 86 内核资源账全等；**229656 条 SASS 逐条全等** |
| 位一致闸（源码重建） | **通过** | 56 s 重建，83 内核资源账全等、0 不同；served 两件各 4096 条 SASS IDENTICAL |
| 路 B：同形状压寄存器天花板 | **否决（编译期）** | `.minnctapersm 3/4` → REG 255→168/128，但 **STACK 0→432/656 B** |
| 新方向：G×NT=16 不变线 | **预筛通过，实测见 §5** | G=4/NT=4 **STACK:0**、SHARED 33920→17664 |
| 迭代成本 | **3–10 min 重建引擎 → 66 s 编译 + 换 cubin** | 见 §4 |

**撤回一个原计划**：原判据「未改动自建 cubin 须与 kit cubin 整文件 IDENTICAL」**不可达且不必要**。ptxas 无法复现 kit 的 `.debug_*` 段（kit 49554312 B vs 重建 11695008 B）。有意义的判据是**指令与资源一致**，本轮据此立闸并通过。整文件 sha256 仅作记录，不作判据。

---

## 1. 位一致闸：两条独立路径都通过

### 1.1 PTX 往返（绕开 C++ 前端）

kit cubin 内嵌完整 PTX 文本：`.nv_debug_ptx_txt`，offset `0x62dc8d`、size `0x17b8d25`（24874277 B），**NUL 是行分隔**。按 ELF section header 抽出、NUL→换行还原，得 `.version 9.3` / `.target sm_121` / `.address_size 64`，**87 个 `.entry`**、665184 非空行。

注意 `cuobjdump -ptx` 对 cubin 返回 0 字节（cubin 只含 SASS，那份 PTX 是 debug 文本），所以必须走 section 抽取。

`ptxas -arch=sm_121`（默认 -O3）重汇编 **20.7 s**，结果：

```
VERDICT RESUSAGE_IDENTICAL (86 kernels)
VERDICT SASS_IDENTICAL (229656 instrs, all kernels)
HI=16 VERDICT IDENTICAL (4096 instrs)   REG:255 STACK:0 SHARED:33920
HI=10 VERDICT IDENTICAL (4096 instrs)   REG:255 STACK:0 SHARED:33920
```

复现：`tools/cubin-roundtrip-gate.sh`（容器内跑，kit 挂 `:ro`，不改 kit）。

### 1.2 源码重建（.cu → cubin）

编译坐标全部**读自源码，非猜测**：

| 来源 | 内容 |
|---|---|
| `src/tensorfold/cuda/exl3/experts.py:129` | `extra_cuda_cflags=["-O3","-lineinfo"]`，sources = experts.cpp/experts.cu/experts_cb{0,1,2}.cu |
| `src/tensorfold/cuda/build.py:33` | `-gencode=arch=compute_{maj}{min},code=sm_{maj}{min}` |
| 容器内 `torch.utils.cpp_extension.COMMON_NVCC_FLAGS` | 4 个 `-D__CUDA_NO_*` + `--expt-relaxed-constexpr`（实测打印，与手写一致） |
| `tools/zig/ops_build.py:22` | 算子 oracle 要求 nvcc **V13.3.73**；容器 nvcc 正是 13.3.73 |

注意 `ops_build.py` 的 `--fmad=false --ftz=false` **只属于算子 oracle**，exl3 扩展不用；混用会破坏一致性。

只编 `experts_cb2.cu`（我方 CB=2）即可，`-cubin` 不需要 pybind 模块，故 `experts.cpp` 可不参与。**56 s**，91 内核：

```
RESUSAGE same=83 diff=0 not-in-kit=8
HI=16 VERDICT IDENTICAL (4096 instrs)
HI=10 VERDICT IDENTICAL (4096 instrs)
```

8 个「仅我方有」是 `m8_expand_kernel` 与 4 参数版 `grouped_mma2_kernel`——我方源码 `c99083245` 比 kit 新，多出的实例化。反向差集 3 件全是 3 参数版 `grouped_mma2_kernel`，**本 loader 不选它**（§4.1 已逐类核对）。

复现：`tools/exl3-source-gate.sh`。

### 1.3 两次假阳性，都被非空断言拦住

**必须记下的教训**，与手册 §7.5 的 `grep -c BUILD_OK` 同一类：

1. 第一版闸报 `RESUSAGE_IDENTICAL` / `SASS_IDENTICAL`，但**所有比对文件都是 0 字节**。根因：`cuobjdump -res-usage` 的 `Function` 行有**前导空格**，`awk '/^Function /'` 永不匹配。
2. 第二版 SASS 取数仍空。根因：传给 `-fun` 的 mangled 名**截断了签名后缀**；补全后又发现从 res-usage 取的名字**带尾随冒号**（`Function NAME:`）。

**现在脚本对每个比对输入强制 `test -s` / 数量下限，空即 FAIL，绝不判 IDENTICAL。** 这条已并入手册 §7.5。

---

## 2. 路 B 否决：压寄存器天花板必然 spill（编译期，20 s）

PTX 层面证实了 E-SELF-03 的源码结论：我方 served entry **只有 `.maxntid 128`，没有 `.minnctapersm`**；而 decode 那 9 个 `grouped_cp_kernel` 变体有 `.minnctapersm 4`（全文件分布：`.maxntid 128`×42、`.maxntid 256`×29、`.minnctapersm 4`×9、`1`×8、`2`×3）。

GB10 占用率算术（65536 regs/SM、48 warps/SM、128 线程/block = 4 warps）：

- 2 blocks → 65536/256 = **256 regs/thread**，现状 REG:255 恰好贴顶，8 warps = 16.7%
- 3 blocks → 需 REG ≤ 168（12 warps = 25%）
- 4 blocks → 需 REG ≤ 128（16 warps = 33.3%）

只给我方两个 entry 注入 `.minnctapersm`，其余 84 内核不动（实测 `OTHERS_UNCHANGED`）：

| 变体 | REG | STACK (HI=16 / HI=10) | 全内核 LDL / STL |
|---|---:|---:|---:|
| kit 基线 | 255 | **0 / 0** | 190 / 495 |
| `.minnctapersm 3` | 168 | **432 / 184 B** | 423 / 715 |
| `.minnctapersm 4` | 128 | **656 / 432 B** | 703 / 1423 |

LDL/STL 增量全落在我方这两件上。**机制**：`acc[G][NT][2][4]` = 2×8×8 = **128 个 fp32 累加器**在此形状下不可压缩，强行降到 168/128 就把累加器/流水状态挤到 local memory，而 local 流量正压在最内层 k 循环的关键路径上。

这正是用户此前提出的「改形状→流量压力→spill→变慢」——**在这一极它是真实成立的**（而在路 A 那两个变体上 STACK 全 0，机制不适用）。

**代价：20 秒编译，零 GPU 时间，零重启。** 复现：`tools/ptx-minblocks-probe.sh`。

---

## 3. 新方向：G×NT=16 不变线

### 3.1 为什么这一极此前没被碰过

`grouped_rows_kernel<CB,NT,W,PF,LO,HI,G,FOLD>`，源码注释（`:403`）明写「a program per (group of G member tiles, …), **the group's weights decoded once**」。所以解码摊销按 G 走：

- 每程序行数 = 16·G，**MG = ceil(maxm/(16·G))** 就是同一份权重被重复解码的倍数
- 总 tile-decodes = (N/16) × KT × **MG**
- 程序总数 = (N/(NT·16)) × MG

**沿 G·NT = 16 这条线**：NT 减半则列块数加倍，MG 减半则成员组数减半，**两者恰好抵消** ⇒ 程序总数不变；同时 `acc` = G·NT·8 = **128 寄存器恒定**，shared 归约元素数 G·NT·16 也恒定。**唯一变化的是解码冗余，随 G 成比例下降。**

路 A 测的两个点都在这条线之外：`G=1,NT=8`（G·NT=8）使 MG 64→128，解码**翻倍**；`G=2,NT=4`（G·NT=8）MG 不变但程序数翻倍，每程序固定开销翻倍。**两者都是增加程序数**，难怪都掉吞吐。

**代价随 G 上升**：warp 归约每个 g 两次 `__syncthreads()`，即每程序 2G 次 barrier。所以存在最优 G，需要实测。

### 3.2 真实形状（读自已有审计，非推算）

`runs/forge-20261123-shape-audit`：down grid `[1,40,820]`、gateup `[1,18,1204]`。
⇒ down 的 N = d = 40×128 = **5120**；gateup 的 N = i = 18×64 = **1152**。
⇒ down：K = 1152，KT = 72，SK=1、W=4 ⇒ 每 warp **18 个 k-tile**。

**18 不是 MMA_KB=4 的整数倍**，这正是 `exl3_experts.zig:207` 那句 `if (… chain_d % 4 == 0) return error.UnsupportedShape; // down would take mma` 放行的原因——**down 走 rows 是形状约束，不是随意选择**。gate-up 的 chain_gu = 320/(4×4) = 20 可被 4 整除，故走 mma。

`grouped_down3_kernel` 虽是专为 down 写的 mma3 结构（上游注释自报 2.9 vs 3.6 ms），但输出 bf16 `Y[P,D]` + 融合 epilogue + 需换 `combine_y`，**非 drop-in**，本轮不动。

### 3.3 编译期预筛（17 s）

`tools/probe-rows-gn.cu`，全部 CB=2/W=4/PF=1/FOLD=0/LO=2：

| NT | G | PF | HI | REG | STACK | SHARED |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 2 | 1 | 16 | 255 | **0** | 33920 |
| **4** | **4** | **1** | **16** | **255** | **0** | **17664** |
| 4 | 4 | 1 | 10 | 250 | **0** | 17664 |
| 4 | 4 | 2 | 16 | 255 | 8 | 17664 |
| 2 | 8 | 1 | 16 | 255 | 72 | 9728 |
| 1 | 16 | 1 | 16 | 255 | 528 | 6144 |

第一行**精确复现基线**，再次印证构建路径忠实。**只有 G=4 零 spill**；G=8 轻微 spill（48/72 B）可作第二臂；G=16 spill 528 B，预筛即淘汰。

---

## 4. 部署路径：一次构建，env 翻臂

### 4.1 loader 契约逐类核对

`exl3_experts.zig:98-116` 按 mangled 名的模板参数选内核，`range()` 是**分桶函数**（`(8,8)→0`、`(2..10)→1`、其余→2），不是数值区间。`tools/loader-coverage.py` 核对三类：

```
mma  <CB,LO,HI,FOLD1,KB4>: kit=3 mine=3   全部 present
rows <CB2,8,4,1,LO,HI,2,0>: kit=3 mine=3  全部 present
cp   <CB2,8,4,S3,LO,HI,EPI>: kit=6 mine=6 全部 present
COVERAGE_OK
```

### 4.2 换 cubin 不重建引擎

- `native.zig:214` `TF_DS_KIT` 指向 kit 目录，引擎启动时读 `cubins/experts_cb.cubin`。
- kit 哈希校验**只在 `prepare.sh:231`**，`start.sh` 不校验；`nodes.sh` 的 `worker_files_check` 只查存在/可读/非空，**不比对尺寸或哈希**。
- `FingerprintMismatch`（`rank_wire.zig:1086`）是**跨 rank 一致性**校验，不是对存储金标准的校验，**不作废权重 rank cache**。
- worker 的 `dsv41-1017-exl3-hf` 卷是 NFS：`addr=192.168.100.10, device:/home/ztxjzh/.cache/huggingface, ro` ⇒ **cubin 只需在 head 换一次**，两 rank 指纹天然一致（实测 worker 侧 sha256 与 head 相同）。
- `start.sh:291` 的 env 直通白名单含 `TF_DS_*`（排除 `TF_DS_KIT` 等），故 **`TF_DS_EXPERT_G` 自动直通两 rank**。

### 4.3 超集 cubin：两臂共用同一份 bit

`tools/experts_cb2_gn.cu` = 未改动的 `experts_cb2.cu` + G=4/NT=4 与 G=8/NT=2 的显式实例化（两个 k2 区间各一件）。66 s，95 内核。

**关键验证：加入实例化后基线 served 两件仍与 kit 逐条 IDENTICAL（各 4096 条）。** 所以对照臂跑的就是今天的 bit，A/B 只翻 env，**不需要在臂间换 cubin**。

已安装：head `experts_cb.cubin` 55172984 B（sha `f4d157b9…`），原件备份 `experts_cb.cubin.kit-orig` 49554312 B（sha `cdb154ba…`）。镜像 `tensorfold-dsv41:base-gn` 两节点均已构建。

### 4.4 引擎侧改动（3 处 + 2d 一致性）

`runs/forge-20261124-self-rows-gn/src/` 存改动后全文。新增 `rowsG()/rowsD()/rowsNt()`（env 读一次即缓存，避免 work list、scratch 尺寸与 launch 三者不一致），替换：

| 文件 | 位置 | 改动 |
|---|---|---|
| exl3_experts.zig | loader 筛选 | `t[1]!=8 … t[6]!=2` → `t[1]!=rowsNt() … t[6]!=rowsG()` |
| exl3_experts.zig | Scratch.sizes / n_d / i32a | `rows_rows_d` → `rowsD()` |
| exl3_experts.zig | down grid | `d/128` → `d/(rowsNt()*16)` |
| exl3_experts2d.zig | 同上 3 处 | 一并改，避免 2d 路径拿到形状不符的内核 |

`rows_rows_d=32` 常量保留（基线默认值语义）。`config()` 的 `chain_d` 只用 W 与 SK，不用 NT，故 `:207` 的门禁不受影响。

构建 `BUILD_EXIT=0`、`tail -1 = BUILD_OK`，并按 §7.5 **验产物内容**：两个二进制 `strings` 均含 `TF_DS_EXPERT_G`。

---

## 5. 实测（同场 A/B/A/B）

见 `raw/ab-gn.log`。口径：未缓存输入 token / `server.prefill_seconds`，28000 字符冷前缀（≈10752 token），每臂两次探针取第二次。

### 5.1 结果：G=4 否决，−1.84%

| 臂 | G | NT | 未缓存 tok | prefill s | tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 8 | 10754 | 5.1144 | **2102.68** |
| 2 | **4** | **4** | 10753 | 5.2296 | **2056.18** |
| 3 | 2 | 8 | 10753 | 5.1449 | **2090.01** |
| 4 | **4** | **4** | 10750 | 5.2205 | **2059.20** |

配对差：臂2−臂1 = **−46.50**（−2.21%）、臂4−臂3 = **−30.81**（−1.47%）。**2/2 同向负**。均值 G=2 2096.35 vs G=4 2057.69 ⇒ **−1.84%**。
每臂第一次探针（未取作读数）也同向：1937.71→1892.42（−45.29）、1930.91→1895.07（−35.84）。

对照臂 2096.35 与权威地板 2084.42 同量级（同场重测，非复用历史值），确认换镜像 + 换超集 cubin 本身性能中性。

### 5.2 根因修正：漏算了激活侧流量

**§3.1 的推导不完整。** 沿 G·NT=16 确实有「程序数不变、acc 寄存器不变、解码冗余 ∝1/G」，但还有一项随之变化：

- 权重侧（解码 ALU + 权重字节）总量 ∝ **MG = maxm/(16G)** ⇒ G 加倍则**减半**
- **激活侧 X 读取总量 = 列块数 × maxm × K，列块数 = N/(NT·16)** ⇒ NT 减半则**翻倍**

所以这条线真正的交换是「**权重侧 ÷2 换激活侧 ×2**」。实测 −1.84% ⇒ **激活搬运侧的边际代价高于造币 ALU 侧**。

**这否证了 E-SELF-03 §7 把「造币工序」当作*绑定*约束的结论**：把解码工作量减半（零 spill、程序数不变）并不能换来吞吐，说明解码 ALU 不是当前的绑定资源。§7 关于「64 个程序重复解码、L2 缓存原料不缓存成品」的**事实描述仍然成立**，但它不是卡点。

与路 A 两个否决值的分解对照（**跨场，仅作机制交叉核对，不作 A/B 判据**）：

| 变体 | G·NT | 解码量 | 激活量 | 实测 |
|---|---:|---:|---:|---:|
| G=1, NT=8（路 A） | 8 | ×2 | ×1 | −0.92% |
| G=2, NT=4（路 A） | 8 | ×1 | ×2 | −2.38% |
| **G=4, NT=4（本轮）** | **16** | **÷2** | **×2** | **−1.84%** |

若两效应近似可加，本轮预期应落在「−2.38% 加一点解码收益」附近，实测 −1.84% 与之一致，且**激活侧代价约为解码侧的 2.6 倍**。

**这个分解必须同场验证**（路 A 是另一场、另一镜像，按纪律不得复用其数值作判据）。§5.3 做同场 2×2 因子实验。

### 5.3 同场 2×2 因子实验：激活侧是绑定约束

为把两个因子分开，把 NT 从 G 解耦（新增 `TF_DS_EXPERT_NT`，默认 16/G，约束 G·NT≤16），
并补编 (NT=4,G=2,PF=1) 这一格（kit 只有 PF=2 版）。超集 cubin 97 内核，基线两件仍逐条 IDENTICAL。
每个测试格前后各夹一次同场基线，`tools/ab-factorial.sh`。

| 臂 | G | NT | 解码量 | 激活量 | G·NT | prefill s | tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 8 | ×1 | ×1 | 16 | 5.1213 | **2099.47** |
| 2 | 1 | 8 | **×2** | ×1 | 8 | 5.1717 | **2078.63** |
| 3 | 2 | 8 | ×1 | ×1 | 16 | 5.1411 | **2091.79** |
| 4 | 2 | 4 | ×1 | **×2** | 8 | 5.2444 | **2050.19** |
| 5 | 2 | 8 | ×1 | ×1 | 16 | 5.1126 | **2102.83** |
| 6 | 4 | 4 | **÷2** | **×2** | 16 | 5.2474 | **2048.82** |

基线三次 2099.47 / 2091.79 / 2102.83，**极差 11.04 tok/s = 0.53%**，测量稳定。

**单因子效应（对相邻同场基线）**

| 因子 | 对照 | 差值 | 相对 |
|---|---|---:|---:|
| 解码量 ×2（激活不变） | 臂2 vs 臂1&3 均值 2095.63 | **−17.00** | **−0.81%** |
| 激活量 ×2（解码不变） | 臂4 vs 臂3&5 均值 2097.31 | **−47.12** | **−2.25%** |
| 解码 ÷2 且激活 ×2 | 臂6 vs 臂5 2102.83 | **−54.01** | **−2.57%** |

**判决**

1. **激活侧边际代价约为解码侧的 2.8 倍**（−2.25% vs −0.81%，同为「翻倍」口径）。
2. **解码量减半几乎不带来收益**：若两效应可乘且对称，臂6 应为 (1−0.0225)×(1+0.0082) ≈ **−1.45%**，
   实测 **−2.57%**；若「解码减半收益≈0」则预期 −2.25%，与实测相差 0.32%（在 0.53% 噪声量级内）。
   ⇒ **数据支持「减少解码量无收益」，不支持对称可加模型。**
3. 因此 **E-SELF-03 §7 把「造币工序（解码 ALU/洗牌）」定为绑定约束的结论被否证**。
   §7 的事实描述（64 程序重复解码、L2 缓存原料不缓存成品、每 MMA 摊 2 次洗牌 + 6 ALU）**仍然成立**，
   但它**不是卡点**——把它减半换不到吞吐。

### 5.4 由此得到的方向修正

绑定约束在**激活 X 的重复读取**：X 读取总量 = 列块数 × maxm × K，当前列块数 = N/(NT·16) = 5120/128 = **40**，
即每份激活被读 40 遍。而权重字节已有 L2 84.5% 命中兜底。

所以 E-SELF-04 的结构改造**目标要换**：不是「把解码后的成品缓存进 shared」，
而是**「把激活缓存进 shared，让一个程序覆盖更多列」**——把列块数从 40 降下来。

已知的量化障碍（下一步要先解的）：
- 一个程序覆盖 32 行 × 全 K 的激活 = 32×1152×2 = **73728 B**，超过 2 blocks/SM 的 51200 B 预算；
  1 block/SM（102400 B）放得下，但占用率降到 4 warps = 8.3%。
- 直接加大 NT 到 16 走不通：`red[W][16][NT*16]` = 4×16×256×4 = **65536 B**，同样超 2 blocks 预算。
- 参照：decode 的 `grouped_cp_kernel` 在 REG:128 下达到 **82.8% DRAM 屋顶**，
  说明低占用率本身不必然低吞吐——若结构改成带宽受限，8.3% 占用率也可能是对的。

**本轮不做该重写**，仅把方向与障碍量化落档。

---

## 6. 边界与未做

- 全部测量为**空载单流**。生产多路（head 最小空闲 4.8 GiB）下**未复验**。
- 本轮只动 down 的 rows 路径形状；**E-SELF-04 原定的「分工造币 + 成品入 shared 双缓冲」结构改造未做**。G×NT 线是它的便宜前置：先用形状把解码冗余降下来，再决定是否值得做结构重写。
- `grouped_down3_kernel`（bf16 + 融合 epilogue + combine_y）未端口。
- 4000 tok/s 目标：Amdahl 上界仍是专家核心归零 2815.83、core+全部 idle 归零 3604.90，**单靠专家核心达不到**，目标是否调整用户未表态。
- 每请求固定成本 275 ms（E-SELF-05 副产）仍未归因。
- 时钟杠杆永久关闭（用户主动降频，升频实测无收益）。

---

## 7. 复现

```bash
# 位一致闸（容器内，kit 只读）
docker run --rm --entrypoint bash -v ~/.cache/huggingface:/hf:ro -v ~/ptxwork:/work \
  tensorfold-dsv41:exl3-tp2-8348338a8d90 /work/cubin-roundtrip-gate.sh
# 源码侧闸
… /work/exl3-source-gate.sh
# 路 B 编译期否决
… /work/ptx-minblocks-probe.sh
# G×NT 预筛
… /work/run-gn.sh          # 源 tools/probe-rows-gn.cu
# 超集 cubin + 验证
… /work/build-gn-cubin.sh ; … /work/verify-gn.sh
# loader 覆盖
python3 tools/loader-coverage.py res-kit.txt res-cb2.txt 2 3
# 部署 + 同场 A/B
bash tools/deploy-gn.sh ; bash tools/ab-gn.sh ~/ab-gn.log 2 4 2 4
```

原始数据：`raw/{gate3.log,srcgate.log,mb.log,res-kit.txt,res-rebuilt.txt,res-cb2.txt,res-gn.txt,res-gn-cubin.txt}`。