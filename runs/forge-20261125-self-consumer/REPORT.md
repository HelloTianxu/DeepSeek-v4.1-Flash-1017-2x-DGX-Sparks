# E-SELF-07：从消费者 SM 回溯，定位当下卡口（离线单内核 stall 分解）

**日期** 2026-11-25，Asia/Shanghai。
**用户指令原话**：「现在的目的是要先把内存吃满，或者把算力吃满，达到硬件瓶颈……思路是从消费者为起点开始找」
「就像做西红柿炒鸡蛋，是西红柿送不上来，还是鸡蛋送不上来？鸡蛋送不上来是因为车运不上来，还是蛋厂没有蛋？」
「全局三项都不满载，本身就是上游有单点卡住的症状，不是三个并列结论。」

**本轮方法论纠正**：此前把「DRAM 49.3% / SM Issue 36.2% / Tensor 37% 三项都不高」当成并列清单，
这是错的。正确做法是从 SM 起沿**一条因果链**逐环回溯，直到找到那个饱和或结构串行的环节。

---

## 0. 结论速览

1. **建立了离线单内核回放能力**（此前缺的就是这个）：只造一层专家权重 **2.96 GiB**，
   不加载整模型，故 ncu kernel replay **内存安全**（驻留态 replay 曾打爆内存需人工拔电源）。
   保真度：gate/up **13.665 ms** vs 生产满块中位 13.3761（**+2.2%**）；down **6.381** vs 6.6802（**−4.5%**）。
   内核身份、grid、block 与生产逐字一致：`grouped_mma_kernel<2,2,10,1,4>` grid (1,18,1204) block 256。

2. **两个最大行的卡点不同，且都不是「卡车没货」**：
   - **gate/up（占窗口 25.79%）**：warp 延迟 barrier **31.4%** + wait **16.3%**，
     等内存（long_scoreboard）只 **12.2%**。
   - **down（12.59%）**：long_scoreboard **35.9%** 居首 —— 这一行确实在等内存。

3. **DRAM 流量零浪费，但被钉在屋顶的 55–63%**：
   gate/up miss **1.998 GB** vs 强制字节 **1.996 GB**（比值 1.001）；
   down miss **0.886 GB** vs 强制 **0.884 GB**（1.002）。
   **重读全部被 L2 吸收，没有一个字节的多余 DRAM 流量。** 达成带宽 155.2 / 135.4 GB/s。

4. **卡口已由 ncu 直接点名：寄存器把占用率钉在 2 blocks/SM。**
   - gate/up：`occupancy_limit_registers=2`、`occupancy_limit_shared_mem=2`（REG 106、SHARED 44288 B）
   - down：`occupancy_limit_registers=2`（shared 允许 3）（REG 255、SHARED 32896 B）

5. **三项候选被实测排除**（见 §4）：死 warp 拖累 barrier、每 warp 的 K 活太少、并发度不足。

6. **已识别但本轮未做的杠杆**：kit 内 **mma2/mma3 世代把权重也改成 cp.async 暂存**，
   我方 loader 选的是最老的 v1（权重走寄存器中转）。见 §6。

---

## 1. 为什么必须离线：把「不能测」变成「能测」

E-DRAM-01 的边界写得很清楚：「具体 stall 原因需要 stall-reason 计数器，PM 采样给不了；
驻留态又禁止 kernel replay」。**这句话是本轮之前所有归因失败的根源** —— 缺的不是分析，是读数。

破法：stall reason 只需要**一个内核**在真实形状下跑，不需要整个模型。
一层专家权重 = 385 专家 × 3 投影 × 2.212 MB = **2.55 GB**，加激活/输出共 **2.96 GiB**。
停服后可用 118 GiB，replay 的 save/restore 完全放得下。

**编译**：kit 公开源码 5 文件（`experts.py:129` 配方），`MAX_JOBS=5`，**78.5 s**，`BUILD_OK`，24 个导出。

---

## 2. 形状与实参：全部反解自生产 trace，非假设

| 证据 | 值 | 反解 |
|---|---|---|
| `gateup_epilogue` grid.x | 14336 | P = R×slots |
| `down_combine` grid | (2048,40,1) | R=2048、D=40×128=**5120** |
| `grouped_mma` grid.y | 18 | I = 18×64 = **1152** |
| `grouped_mma` grid.z | 1204 = 2×602 | n_gu=602 ⇒ maxu=**385** |
| `grouped_rows` grid.z | 820 | = listLen(14336,385,32) ✓ |

`listLen(p,maxu,rows)=(p+maxu*(rows-1))//rows`；`maxm=memberStride(2048)=2048`。

**k2 实测**：checkpoint safetensors 头，layer3 的 384 专家 × w1/w2/w3 **全部 k2=6**（3.0 bpw），
形状 w1/w3 (320,144,48)、w2 (144,320,48) —— 全宽 I=2304，TP2 切半后本地 1152 ✓。
`range(6,6)` ⇒ 桶 1 ⇒ 内核 `<2,10,…>`，与 ncu 报出的 `<2,2,10,1,4>` 一致。

**权重寻址**（`experts_grouped.cuh:178/597`）：`T + (kt*NTILES+nt)*TW + lane`，`TW=4*K2=24` words
⇒ 每 tile 96 B，每专家每投影 `320*72*24*4 = 2211840 B`。

**一个必须记住的 API 细节**：wrapper 里 **`B0/B1` 是 k2 数组**（`a.k2_0=B0.data_ptr<int>()`），
**`TP0/TP1` 才是 int64 指针数组**。猜反则全盘皆错。

**工作列表**用导出的 `m2_work`（`experts.cu:452`）构造，排序为**专家 place 升序、组序升序**，
超出活跃数的项 `u=maxu` 被内核早退 —— 与生产 `work_list` 同语义。

---

## 3. 因果链读数（ncu，`--clock-control none` 保持用户实际时钟）

### 3.1 第一环：SM 为什么发不出指令

warp 延迟按原因分解（`smsp__average_warp_latency_issue_stalled_*`，只用 `.pct` 归一）：

| 原因 | gate/up | down | 含义 |
|---|---:|---:|---|
| **barrier** | **31.4%** | 4.9% | 等块内兄弟 warp 到栅栏 |
| **wait** | **16.3%** | 15.3% | 等固定延迟的执行依赖链 |
| **long_scoreboard** | 12.2% | **35.9%** | 等全局/L2 数据 |
| selected | 10.1% | 15.2% | 正在发射（非 stall） |
| math_pipe_throttle | 9.5% | 7.3% | 执行管线满 |
| short_scoreboard | 6.8% | 10.0% | 等 shared/MIO |
| not_selected | 6.4% | 3.7% | 就绪但发射口忙 |
| mio_throttle | 4.0% | 2.9% | MIO 队列满 |
| 其余合计 | 3.3% | 4.8% | |

### 3.2 第二环：那一环饱和了吗 —— 都没有

| 量 | gate/up | down |
|---|---:|---:|
| SM issue（% peak active） | 41.20 | 33.05 |
| ALU heavy | 29.89 | 26.55 |
| FMA heavy | **33.55** | 30.06 |
| Tensor（HMMA） | 28.34 | 32.44 |
| L2 吞吐（% peak elapsed） | **44.96** | 27.10 |
| L1TEX 吞吐 | 43.53 | 27.14 |
| 达成占用率 | 33.19 | 16.58 |

**没有任何一条管线接近饱和，最高只有 L2 的 45%。** 所以「蛋厂满负荷」也不成立。

### 3.3 第三环：卡车（DRAM）—— 零浪费，但只跑到 63%

| 量 | gate/up | down |
|---|---:|---:|
| 时长 | 12.8718 ms | 6.5442 ms |
| L2 读请求 | 9.072 GiB → **756.8 GB/s** | 2.377 GiB → 390.0 GB/s |
| L2 读 miss（DRAM 上界） | 1.861 GiB → **155.2 GB/s** | 0.825 GiB → 135.4 GB/s |
| L2 读命中 | **79.5%** | 65.3% |
| 占 246.2 GB/s 屋顶 | **63.0%** | 55.0% |
| **强制字节**（权重+激活） | **1.996 GB** | **0.884 GB** |
| **实测 miss / 强制** | **1.001** | **1.002** |

强制字节算法：gate/up = 385×2×2.212 MB + 2×(14336×5120×2 B) = 1.702+0.294 = 1.996 GB；
down = 385×2.212 MB + 14336×1152×2 B = 0.851+0.033 = 0.884 GB。

**⇒ 重读一个字节都没落到 DRAM。E-ROOF-01 的 H1（已在带宽屋顶、只能消除重读）彻底不成立。**

### 3.4 第四环：指令流成分 —— MMA 只占 4.3%

| 类别 | gate/up | 占比 |
|---|---:|---:|
| 总 warp 指令 | 2,285,455,680 | 100% |
| shared_ld | 78,833,664 | 3.45% |
| ldsm（ldmatrix） | 24,560,640 | 1.07% |
| **global_ld** | **20,264,832** | **0.89%** |
| shared_st | 19,199,232 | 0.84% |
| branch | 99,425,520 | 4.35% |
| **MMA（按执行行数推算）** | **≈98.2 M** | **4.30%** |

**95.7% 的指令不是 MMA，而 load 只占 0.89%。**
SM 把发射槽几乎全花在「造币」（解码/寻址/洗牌）上，**递货的手太少**，
所以卡车空着 —— 不是路不好，是没人往车上装。

### 3.5 第五环：谁把 warp 数钉住了 —— ncu 直接点名

| 限制资源（blocks/SM） | gate/up | down |
|---|---:|---:|
| **registers** | **2** | **2** |
| shared_mem | **2** | 3 |
| warps | 6 | 12 |
| blocks（硬上限） | 24 | 24 |
| REG/thread | 106 | 255 |
| SHARED/block | 44288 B | 32896 B |

**寄存器是两个内核共同的绑定约束**（gate/up 上 shared 并列）。
2 blocks ⇒ gate/up 16/48 warps = 33%，down 8/48 = 16.7%，与实测占用率吻合。

---

## 4. 三项候选被实测排除

### 4.1 「补齐行导致空转 warp 拖累 barrier」—— 否证

真实路由（6 routed + 末位共享专家）下：ucount=385，共享专家独占 **2048 行**，
routed 中位 32、最冷 17、最热 50。gate/up 活跃组 **416**/容量 602，down **617**/820。

**FILL 对照**：把每专家末组补满 64 个有效 member（重复已有 code），
**ucount、组数、权重字节全不变**，只让原本空转的 warp 变活跃。
结果 gate/up **13.665 → 14.450 ms（+5.7%，变慢）**。
⇒ 空转 warp 不是负担，它们本来就该闲着；**barrier 读数被「死 warp 停在栅栏上白等」污染了**，
不能直接判 barrier 为可动卡口。

**零空转对照 full64**（224 专家各恰好 64 行）：gate/up 8.354 ms。
权重字节比 0.644、时长比 0.611 —— **两者几乎相等**；
换算达成带宽 153.8 GB/s，与 shared6 的 155.2 **几乎相同**。
⇒ **达成带宽被钉在 ~154 GB/s，与工作成分无关。**

### 4.2 「每 warp 的 K 活太少，ILP 不够」—— 否证

SK 单因子扫描（WK 固定 4，SK×WK 整除 KT=320 且 chain%4==0，保证同一 KB=4 变体）：

| SK | chain | gate/up 中位 |
|---:|---:|---:|
| 4（生产） | 20 | 13.4265 / **12.7419**（同配置两次） |
| 1 | 80 | 13.2293 |
| 2 | 40 | 12.8932 |
| 5 | 16 | 12.9445 |
| 8 | 10 | 内核拒绝（chain 10 无允许步长） |
| 20 | 4 | 13.7871 |

**全部落在基线自身噪声带 12.74–13.43（5.4%）内 ⇒ 无效应。**

### 4.3 「并发度不足，加并行就能顶高带宽」—— 否证

同会话内串行 3 次 vs 3 条流并发 3 次：
gate/up 39.0090 → 38.9620 ms（**1.001×**）；down 19.1713 → 19.1699（**1.000×**）。
⇒ **完全零收益。grid 早已填满机器**（mma 21672 个程序 vs 48 SM×2 blocks=96 并发位），
卡的不是并发度，而是**每 SM 能驻留多少 warp**。

---

## 5. 本轮我自己犯的两个错（已当场纠正）

1. **又把容量当活跃量**：先算出 gate/up「补齐 1.857×」并差点当成执行浪费。
   实际 warp 按 16 行一组、`rows_sh[rg*16]<0` 整组跳过，
   **执行行数 = 17056 / 14336 = 1.190×**，不是 1.857×。
   1.857 是**容量上界**。这是 E-SELF-04 §8.1 同一类错误的复发，已在工具里显式区分两个口径。

2. **差点用 GB10 的零计数器下结论**：`ldgsts_cache_access = 0`，
   我据此宣称「两个内核都没有 cp.async」。**错。**
   源码 `experts_grouped.cuh:613/645` 明确对**激活**用了 `cp_async16`。
   这正是 E-DRAM-01 §1 记过的陷阱（`aperture_device` 恒 0）。
   **准确事实**：激活走 cp.async 环（MMA_STAGES=3，提前 2 步）；
   **权重(trellis)字走「全局→寄存器→shared，提前两步」**（`:558` 注释 + `:597` `load_words`）。

3. **一次假测量被自己拦住**：并发首测报「1033× 加速」，
   根因是 event 记在默认流上、没有 bracket 侧流工作。
   已改为「侧流 wait 起始事件 → 各侧流结束事件被默认流 wait」，重测得 1.001×。

---

## 6. 当下卡口的表述，与已识别的杠杆

**卡口（有读数支撑的部分）**：
从 SM 往回走 —— 发射 41% 不饱和 → warp 主要卡在栅栏与固定延迟依赖 → 各执行管线均 ≤34% 不饱和
→ DRAM 零浪费却只到 63% → **因为 load 只占指令流 0.89%，而在飞请求数受「每 SM 驻留 warp 数」限制**
→ **驻留 warp 数被寄存器钉在 2 blocks/SM**（ncu 直接报出）。
并发、K 活量、死 warp 三项均已排除。

**尚未分离的最后一环**：寄存器同时扮演两个角色 ——
既限制驻留 warp 数，又是**权重暂存的载体**（提前两步的 trellis 字握在寄存器里）。
这两者哪个是主导，本轮未分离。

**已识别的杠杆（本轮未做）**：kit 内更晚的世代把权重也改成了 cp.async：
- `:798` 注释：「Rows and trellis words come through a ring of S stages of KB k tiles by cp.async (one barrier a stage)」
- `:847` 激活、**`:852` trellis 字**，两者都是 `cp_async16` ⇒ `grouped_mma2_kernel`（`:893`，`__launch_bounds__(256,1)`）
- `:1031/:1037` 同构 ⇒ `grouped_mma3_kernel`（`:1311`）
- `grouped_down3_kernel`（`:1417`）属 mma3 系

**我方 loader 选的是 v1 `grouped_mma_kernel`（权重走寄存器中转）。**
把权重暂存搬到 cp.async 可在**不增加寄存器压力**的前提下提高在飞字节数 —— 正好对着上面那个双角色。
`grouped_mma2` 已在编译产物里导出，可离线直接试。
**注意**：kit 独有的 3 件是 3 参数版 `<2,2,10>/<2,6,6>/<2,8,8>`，签名与 v1 不同，需先读 `grouped_mma2` wrapper 与 `m2_work(M2_ROWS)` 契约，不是 drop-in。

---

## 7. 边界

- 只测 head 单卡、单层、单请求形状；**未在生产多路（head 最小空闲 4.8 GiB）下复验**。
- 路由是合成的（真实分布未知）；`shared6` 假设末位为共享专家，`full64` 专家数 224≠385，
  **故 full64 只用于比较带宽结构，不比总时长**。
- 权重位流是随机数：解码 ALU 与访存代价与数值无关，但**未做数值正确性校验**（本轮只测性能）。
- `long_scoreboard` 等读数是全 warp 平均，**含空转 warp**，§4.1 已证其被污染。
- `sectors/request` gate/up **8.45**、down **4.81**（高于 96 B 连续读应有的 3），
  提示访存合并可能不佳，**但未做逐指令归因，不下结论**。
- L2 屋顶仍未测（只知吞吐 45% of peak，peak 的绝对值未标定）。
- **未测 mma2/mma3**，§6 的杠杆是源码事实 + 机制推理，**不是实测收益**。

---

## 8. 复现

```bash
# Spark head，停服后（avail 118 GiB）
docker run -d --name exl3-probe --gpus all --entrypoint bash \
  -v /home/ztxjzh/ptxwork:/home/ztxjzh/ptxwork -e HOME=/home/ztxjzh \
  --cap-add SYS_ADMIN tensorfold-dsv41:exl3-tp2-8348338a8d90 -c "sleep 7200"
docker exec exl3-probe bash -c "cd /home/ztxjzh/ptxwork && MAX_JOBS=5 python3 probe-build.py"   # 78.5 s
docker exec exl3-probe bash -c "cd /home/ztxjzh/ptxwork && python3 probe-layer.py --routing shared6"
docker exec exl3-probe bash -c "cd /home/ztxjzh/ptxwork && bash probe-ncu.sh"      # stall 分解
docker exec exl3-probe bash -c "cd /home/ztxjzh/ptxwork && bash probe-pipes.sh"    # 管线饱和度
docker exec exl3-probe bash -c "cd /home/ztxjzh/ptxwork && bash probe-occ.sh"      # 占用率限制资源
python3 ~/ptxwork/probe-parse.py ; python3 ~/ptxwork/probe-pipes-parse.py
# 对照
docker exec exl3-probe bash -c "cd /home/ztxjzh/ptxwork && python3 probe-layer.py --fill"
docker exec exl3-probe bash -c "cd /home/ztxjzh/ptxwork && python3 probe-layer.py --routing full64"
docker exec exl3-probe bash -c "cd /home/ztxjzh/ptxwork && python3 probe-layer.py --conc 3 --reps 8"
docker exec exl3-probe bash -c "cd /home/ztxjzh/ptxwork && bash probe-sk-sweep.sh"
```

工具已入 git：`tools/probe-{build,layer,parse,pipes-parse}.py`、`tools/probe-{ncu,pipes,occ,sk-sweep}.sh`。
原始证据 `raw/`（11 件）。**生产已恢复，`check-boot RESULT: OK`（avail 10 GiB）。**