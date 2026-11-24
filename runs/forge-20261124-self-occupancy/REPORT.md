# E-SELF-02：提高专家内核占用率 —— 机制成立，但**杠杆不可实施**（kit 二进制-only）

**日期**：2026-11-24
**判决**：**不实施**。不是机制被证伪——**机制被实测证实了**——而是
**我方拿不到可改的对象**：专家内核是 kit 预编译 cubin，无源码；
且 shipped 变体里没有同形状的低寄存器替代件。
**基线不变，仍 53 patch；生产 `base-alias` 已恢复，`check-boot.sh` RESULT: OK（avail 10 GiB）。**

本轮**未改任何 kernel 代码、未导出 patch**。花费 1 次带诊断 flag 的全量构建（已回滚）
+ 3 次只读单内核编译 + 3 次 cubin 只读反汇编。
**在「动手改之前先判物理空间」这一步就判死，省下了一次注定无收益的 A/B。**

---

## 1. 结论先行

| 问题 | 答案（实测） |
|---|---|
| prefill 专家 down 是哪个内核？ | **`tf_exl3x::grouped_rows_kernel<2,8,4,1,2,16,2,false>`**（kit cubin） |
| 它真的 255 regs 吗？ | **是。REG:255、STACK:0（零 spill）、SHARED:33920 B** |
| 静态占用率真的是 16.7% 吗？ | **是。** 255×128=32640 ⇒ 2 blocks ⇒ 8/48 warps = **16.7%**，与 achieved 17% 吻合 |
| 寄存器是限制因子吗？ | **是。** shared 33920 B 允许 3 blocks，寄存器只允许 2 ⇒ **寄存器绑定** |
| 能降寄存器吗？ | **不能。无 .cu 源码，且无兼容的低寄存器变体** |

**⇒ E-DRAM-01 的机制归因（低占用延迟受限、寄存器压力）是对的。
E-SELF-02 的方向在物理上成立，在工程上被 kit 的分发形态堵死。**

## 2. 内核身份：我方走 `tf_exl3x`，不是引擎源码里的 `tf_experts*`

按 mangled 名反查（`grep -rn "_ZN" zig/src/families/dsv41/*.zig`）：

- 我方 dsv41 专家路径 = **`tf_exl3x::grouped_rows_kernel`（prefill down）
  / `grouped_mma_kernel`（prefill gate/up）/ `grouped_cp_kernel`（decode）**，
  全部定义在 **`kit/cubins/experts_cb.cubin`**（49.5 MB 预编译二进制）。
- 引擎仓 `grep -rln "grouped_cp_kernel\|grouped_rows" --include=*.cu --include=*.cuh` ⇒ **0**
- kit 仓 `find . -name "*.cu" -o -name "*.cuh"` ⇒ **0**（kit 只发 cubin）

**附带查明一个家族混淆**：引擎源码里那 8 个
`tf_experts` / `tf_experts_prefill` / `tf_qmm_group` / `tf_qmm_prefill` / `tf_lane_gemv` / `tf_experts_pack`
内核，mangled 名**全部只出现在 `zig/src/families/nemotron/cuda_kernels.zig`**，
**我方一个都不调用**。它们的实测寄存器（`raw/engine-ptxas.txt`）：

| 内核 | regs | threads | spill |
|---|---:|---:|---:|
| `tf_experts::expert_kernel<64,1,0/1,2,4>` | 128 | 128 | 0 |
| `tf_experts_prefill::prefill_kernel<64,1,1/3,2,2,4>` | 98 | 256 | 0 |
| `tf_qmm_group::group_kernel<64,16,64,1,4,8,…>` | 48 | 1024 | 0 |
| `tf_qmm_prefill::prefill_kernel<64,128,128,2,4,3,false>` | 128 | — | 0 |
| `tf_lane_gemv::gemv_kernel<64,64,8>` | 64 | — | 0 |
| `tf_experts_pack::pack_kernel<2>` | 20 | — | 0 |

⚠️ **连带更正 E-SELF-01**：该报告 §2 读 `lane_gemv.cu` / `experts.cuh` / `qmm_group.cu`
得出的「我方已有 cp.async 7–8 级软件流水，强于上游」**依据的是 Nemotron 家族源码**，
不能代表我方 exl3 内核。我方 exl3 流水结构**未知**（无源码）。已加更正批注。

## 3. 静态占用率核算（48 SM / 48 warps/SM / 65536 regs/SM）

served prefill down：`grouped_rows_kernel<2,8,4,1,2,16,2,false>`，W=4 ⇒ **128 threads/block**

- **寄存器**：255 × 128 = 32640 regs/block ⇒ ⌊65536/32640⌋ = **2 blocks** ⇒ 8 warps ⇒ **16.7%**
- **shared**：33920 B/block ⇒ 100 KiB/SM 允许 3 blocks ⇒ **不是限制因子**
- **spill**：STACK:0、LOCAL:0 ⇒ 当前变体**没有溢出**（同族 FOLD=1 变体有，见 §4）

**寄存器绑定，静态 16.7%，achieved 17% —— 两者一致，机制成立。**
若能把 regs 压到 ≤128，可得 4 blocks = 33.3%，在飞请求翻倍。**这正是原计划的收益来源，物理上真实存在。**

## 4. 为什么拿不到这个杠杆：两条都堵死

### 4.1 无源码
kit 只分发 cubin。`__launch_bounds__` 第二参数、staging 深度、寄存器分配**全在编译期**，
二进制到手即固定。**改不了。**

### 4.2 无兼容的低寄存器变体
引擎的选择器（`zig/src/families/dsv41/exl3_experts.zig:104-109`）硬筛：

```zig
// <CB, NT, W, PF, LO, HI, G, FOLD>
if (t[0] != codebook or t[1] != 8 or t[2] != 4 or t[3] != 1 or t[6] != 2 or t[7] != 0) continue;
k.rows[range(t[4], t[5])] = try fnByName(cb, name);
```

即要求 **CB=2, NT=8, W=4, PF=1, G=2, FOLD=0**，只有 LO/HI（K2 范围）可变。
我方 checkpoint 的 down **K2 = 6..10**，故需 range 覆盖 6..10。

cubin 里全部 24 个 `grouped_rows` 变体（`raw/kit-exl3x-named-resusage.txt`）中：

| 变体 `<CB,NT,W,PF,LO,HI,G,FOLD>` | REG | STACK | 引擎可用？ |
|---|---:|---:|---|
| `<2,8,4,1,2,16,2,0>` | **255** | 0 | ✅ **served** |
| `<2,8,4,1,2,10,2,0>` | **255** | 0 | ✅ **served** |
| `<2,8,4,1,8,8,2,0>` | 204 | 0 | ❌ K2 范围 8..8 不覆盖 6..10 |
| `<2,8,4,1,2,16,1,0>` | 165 | 0 | ❌ G=1 |
| `<2,8,4,1,2,10,1,0>` | 168 | 0 | ❌ G=1 |
| `<2,8,4,1,8,8,1,0>` | **121** | 0 | ❌ G=1 且范围 8..8 |
| `<2,8,4,2,2,16,1,0>` | 168 | 0 | ❌ PF=2, G=1 |
| `<2,8,4,2,8,8,1,0>` | 135 | 0 | ❌ PF=2, G=1, 范围 8..8 |
| `<2,8,4,1,2,16,2,1>` | 255 | **1032** | ❌ FOLD=1（且真溢出） |

**满足 `PF=1,G=2,FOLD=0` 且覆盖 K2 6..10 的，只有那两个 255 regs 变体。**
低寄存器的（121/135/165/168）全部 G=1 或范围不符——**G 与 PF 是算法形状参数，
不是可自由替换的调优旋钮**，改选择器会改变语义与 launch 参数布局，不是「换个变体」那么简单。

**⇒ 两条路都堵死，本轮否决。**

## 5. 保留的事实与后续条件

- **机制成立且已量化**：prefill 专家 down 寄存器绑定、静态 16.7%、零 spill。
  压到 ≤128 regs 可得 33.3%——**这是一个真实存在、目前拿不到的收益**。
- **解锁条件（任一）**：① 上游放出 exl3 内核源码；
  ② 上游在 cubin 里增发 `PF=1,G=2,FOLD=0` 且低寄存器的 prefill 变体；
  ③ 我方自写 exl3 2.9bpw codebook GEMM（量级远超本轮，需独立立项）。
- **可向上游提的具体请求**：现有 shipped 变体证明上游会编译多种形状
  （FOLD=1 变体甚至带 1032 B spill），**唯独没有 G=2/FOLD=0 的低寄存器档**。
  这是一个有实据、可复现的 feature request，而不是泛泛的「请优化」。
- **基线不动**：53 patch，生产 `base-alias`，`check-boot.sh` RESULT: OK。
  诊断 flag 已回滚，引擎树与 Spark 构建目录均回到基线。

## 6. 本轮方法论收获（已入手册）

1. **改内核前先确认两件事：它属于哪个家族、我方有没有它的源码。**
   引擎里同名近形的 kernel 可能整个属于另一个家族；exl3 路径是 cubin，无源码。
2. **静态占用率必须从二进制读**（`cuobjdump -res-usage` / `-Xptxas -v`），不能沿用文档数字。
3. **筛变体时不要只筛一个实例化就下结论。** 本轮中途我曾只筛 decode 的
   `grouped_cp_kernel`（128 regs）就宣称「255 不存在、E-DRAM-01 错了」——**那是错的**，
   255 属于 prefill 的 `grouped_rows_kernel`。**全量列出同族变体后再判决。**
4. **「机制成立」≠「杠杆可实施」。** 分发形态（二进制 vs 源码）本身就是一道闸，
   应加入手册的五道闸，成为**第六道：可改性闸**。

## 7. 复现

```bash
bash tools/kernel-resusage.sh            # kit cubin + 引擎 kernel 寄存器报告
KIT_ONLY=1 bash tools/kernel-resusage.sh # 只看 cubin（快，不编译）
```

原始证据：`raw/kit-exl3x-named-resusage.txt`（86 个 tf_exl3x 内核，名+REG+STACK+SHARED）、
`raw/kit-cubin-resusage.txt`、`raw/engine-ptxas.txt`。