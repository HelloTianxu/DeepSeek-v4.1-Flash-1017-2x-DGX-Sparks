# B 前置测算：chunk_rows 变大的内存账与安全评估

**日期** 2026-11-24　**状态** 只读测算完成，**未做任何构建、未改任何配置、未动运行中的服务**
**触发** 用户指令「先测算 评估安全后我们动手」

---

## 0. 一句话结论

> **【最终判定，2026-11-24 修正】chunk 4096 与 8192 均不安全，B 方向否决。**
> 用户给出生产基线：**生产多路上下文、KV 累积时 head 最小空闲 4.8 GiB**，
> 不是本报告初版所用的空载单流 8.22 GiB。扣除 memguard 地板 2.5 GiB，
> **可用余量仅 2.3 GiB**，4096 的结构估计（+3.40）与最坏情形（+3.68）均装不下。
> 另有独立的 **Amdahl 否决**：专家 core 归零上限仅 2815.83 tok/s，低于 4000 目标。
> 详见 §0.1。以下 §0 原表保留作**空载口径**的记录，不作为决策依据。

**（空载口径，已被生产基线取代）** 4096 内存上安全，8192 不安全。

| 档位 | arena 增量区间 | head 加载后 avail | 相对 memguard 地板(2.5 GiB) | 空载判定 | **生产判定(4.8 GiB 基线)** |
|---|---|---|---|---|---|
| **4096** | +1.31 .. +3.68 GiB | 4.54 .. 6.91 GiB | +2.04 .. +4.41 GiB | 安全 | **不安全**：avail 1.12–3.49，结构估计 1.40 **跌破地板** |
| 8192 | +3.94 .. +11.03 GiB | −2.82 .. 4.28 GiB | −5.32 .. +1.78 GiB | 不安全 | **不安全**：avail −6.23 .. 0.86 |

实测起点（本轮 `/proc/meminfo`，**空载单流**）：head `MemAvailable` **8.22 GiB**、worker **9.93 GiB**。
head 是约束方（两 rank 各自分配自己的 arena，worker 余量更大）。

**并且 B 不是单变量改动**：改 chunk 必须同时改 POOL，否则引擎直接拒绝启动。详见 §3。

**补充结论（§2.5 实测）：4096 不会「没空间做推理」。** 推理期间的内存消耗实测只有
**22–40 MiB**，且**不随 prompt 长度增长**。风险全部在**加载期**，不存在第二笔运行时大账。

### 0.1 生产基线修正与 Amdahl 否决

**内存侧**：引擎自报 `121.63(total) − 104.54(model) − 12.2(reserve) = 4.86 GiB`，
与用户实测的生产最小 4.8 GiB 吻合。空载之所以有 8.2 GiB，是因为 KV 池虽在加载期虚拟分配，
统一内存下**物理页按需 fault-in**；生产多路把池填满后真实占用上升。

**Amdahl 侧**（`forge-20261123-bottleneck-ledger/REPORT.md` §3，早已算好）：
专家 core 归零 → 3.064108 s → **2815.83 tok/s**；
专家 core + 全部间隙都归零 → 2.393413 s → **3605 tok/s**。
**两者都低于 4000 目标（需 2.157 s）。chunk 杠杆的数学天花板够不到目标。**

⇒ **双重否决，不再作为首选杠杆。** 转向零内存成本的等待项，
已落地 `ENGRAM_AIO=0`（+19.5%/+20.1%，见 `forge-20261124-engram-aio/REPORT.md`）。

---

## 1. chunk_rows 的真实语义：远不止 prefill 批大小

只读代码考古（`zig/patches/` 52 个 patch + Spark 上引擎源码树）查明：

1. **它是编译期常量，没有 env 开关。**
   `model.zig:38 pub const chunk_rows = 2048;`。注释里写的 `TF_DS_PREFILL_CHUNK`
   在上游 `model.zig:37` 与 `native.zig:92` **同样只是注释**，出厂二进制 strings 里也没有该 env。
   ⇒ 想改必须改 patch + 重建，没有热切换路径。

2. **它同时是 KV 池的块粒度。** `parts.zig`：
   - `pub const chunk_rows: usize = 2048;`（注释：A pool block: model.chunk_rows）
   - `pub const growth_rows: usize = 16 * chunk_rows;`（随之自动变）
   - `lanes.zig` 断言 `parts.chunk_rows == model.chunk_rows` ⇒ **两个常量必须同改**
   - `if (c.cap == 0 or c.cap % (parts * chunk_rows) != 0) return error.BadPoolSize`
   - reservation `alignForward(len + max_new + round_rows + 2, chunk_rows)`
   - 多处 `size % chunk_rows != 0 → BadSize`、`base % chunk_rows != 0 → BadRange`

3. **它进 spill tier identity。** `spill.Geometry.of(c, chunk_rows, …)`；
   tier identity 结构里有 `.chunk_rows` 字段，且自带「改 chunk_rows 则 identity 变」的变异测试。
   ⇒ 现有 64 GiB spill tier 的块会失配，需重建（NVMe 上，不吃 RAM，但影响启动耗时）。

4. **两个现成 env 不能替代。** `0051-prompt-rows.patch` 的
   `TF_DS_PROMPT_WINDOW_ROWS`（默认 64，上限 128）与 `TF_DS_PROMPT_GEMM_ROWS`（默认 128，上限 128）
   只在行数**小于/大于**阈值时切换路径，2048 行主体两者都不触发 ⇒ 无法用它们提高每专家 M。
   且注释明确「两条路径舍入不同，同一 token 的比特可能不同」⇒ 这两个 knob 属**有损**开关。

---

## 2. 内存账

### 2.1 分配公式来源

`prompt.zig` → `Chunk.init(e, a, cap, tokens)`，其中 `cap == model.chunk_rows`。
逐项分配几乎全是 `cap * X` 形式。最大单项是专家 scratch：

```
exl3_experts.Scratch.sizes(rows, slots, d, i, e):
  p = rows*slots;  maxu = min(p, e)
  xg = xu = 2*p*d      xd = 2*p*i      z = 4*promptZ(d,i)*p
  promptZ(d,i) = max(2i + d/2, d)
  members = 4*maxu*ceil128(rows)
  work_gu/d = 8*listLen(p, maxu, 64/32),  listLen(p,m,r) = (p + m*(r-1))/r
```

代入真实配置（`checkpoint-config.json`）：d=5120、hc=4、hd=512、heads=64/world=**32**、
o_groups=8、o_lora=1024、q_lora=1280、experts=385、slots=7、world=2、
index_heads=32、index_head_dim=128、index_topk=512、candidate_blocks=2048、
hc_blocks=40、TP2 本地专家 I=**1152**。

### 2.2 cap=2048 时已枚举的 cap 线性项

| 项 | MiB |
|---|---|
| expert_scratch | **594.5** |
| h + h_alt | 160.0 |
| ga + gm | 160.0 |
| q + o | 128.0 |
| pa + xf + pm | 120.0 |
| u / iq+iq4 / pidx | 32.0 each |
| x | 20.0 |
| cidx + ktop | 16.0 |
| 其余小项 | ~30 |
| **合计** | **1343.3 MiB = 671.6 KiB/行** |

引擎自报 `arena 3766 MiB used` ⇒ 已枚举部分占 **35.7%**。

### 2.3 上下界

- **下界 35.7%**：只算已逐项枚举的（漏掉 engram 的 eb/ek/ekg/kv、grouped-linear、prompt2d.Two 等，它们同样随 cap 线性）。
- **结构上界 ~92%**：arena 里可辨认的**常量**项只有约 284 MiB
  （keys 128 + score 64（`ke=min(cap*max_comp, 2^24)` 已封顶，不随 cap 长）、gate_f 7.5、
  tmax/bmax/gl ~11、blas_ws ~32、ws.w ~42）。其余几乎全是 `cap * X`。
- 真实值应在两者之间、偏向上界。**这个不确定性无法在构建前消除**，因为
  `prompt.Chunk` 完整公式所在的上游 pin 版本源码树在 host 与镜像内都不存在
  （镜像构建时 `rm -rf /tensorfold`；host 只有 ref **f77d5a1** 的树，与生产 pin **fbbc9713** 不同版本）。

⇒ 增量按**区间**给，判定按**最坏情形**做。这正是 §0 表格的算法。

### 2.4 为什么不能靠缩 KV 池腾内存

`parts=2` 时 POOL 必须 ≥ 2×WINDOW。WINDOW=524288 ⇒ **POOL 最小 1048576**。
当前 1052672 = 1048576 + 4096，正好只多一个 chunk 块，**已经几乎贴着下界**。
⇒ 想给 8192 档腾出 4–11 GiB，只能降 CONTEXT，那是产品能力变更，不在本次范围。

### 2.5 运行时账：推理期间到底吃多少（实测，回答「会不会没空间跑推理」）

§2.1–2.4 算的是**加载期**装不装得下。运行时是另一笔账，项目自己有规矩：

- `start.sh` 加载后告警线：`avail < MEMGUARD_GIB + 3` ⇒ 期望 **地板之上留 3 GiB**，
  注释理由「a long prompt's workspace comes out of this headroom」。
- `prepare_gate.zig`：`TENSORFOLD_PREPARE_FLOOR_GIB` 默认 **3 GiB**，
  `per_byte = 24`（每字节文本预留 24 B，host 侧渲染+tokenize）。

按这条 3 GiB 规矩，4096 最坏情形只剩 2.04 GiB，**是踩线的**。所以不能靠推断，实测。

**方法**：200 ms 采样两节点 `/proc/meminfo` MemAvailable，期间发一个冷前缀请求。
不注入引擎、不 replay、memguard 全程在位，单流。脚本 `measure_runtime_dip.sh`。

| 请求 | 未缓存 token | prefill | head 起点→最低 | **head 跌幅** | worker 跌幅 |
|---|---|---|---|---|---|
| 28 000 字符 | 10 755 | 6.06 s | 8.262 → 8.223 GiB | **0.039 GiB** | 0.010 GiB |
| 120 000 字符 | **41 364** | 22.56 s | 8.218 → 8.196 GiB | **0.022 GiB** | 0.043 GiB |

**prompt 长 4.3 倍，内存跌幅没有变大（反而更小）。**

**机制解释**：device arena 在加载期就按 `cap = chunk_rows` 与 `request_cap` **预分配到位**，
`n > chunk_rows → BadChunk` 保证任何 chunk 都不超上限。长 prompt 只是**多跑几个 chunk**，
不再申请设备内存；KV 池同样是预分配的 position 池，reservation 是池内记账而非新分配。
host 侧 prepare 预留 = 24 × 文本字节，120 000 字符也只有 ~2.6 MiB。

⇒ **4096 档最坏情形加载后 avail 4.54 GiB，推理再吃 0.04 GiB，仍高于 memguard 地板 2.0 GiB，
是实测运行时需求的约 50 倍。不会没空间做推理。**

`start.sh` 的「3 GiB」是面向「head 上还跑桌面和其他容器」的**通用保守告警**，
不是本工作负载的实测需求；本次会触发该 warn，但按实测不构成阻塞。

**自我修正**：初版方案提议把 `MEMGUARD_GIB` 2.5 → 4.0，**撤回**。
最坏情形下那样只剩 0.54 GiB 高于地板，正常扰动（页缓存、别的容器）容易触发误杀，
反而降低可用性。上次事故的根因是**手动 docker run 绕过了 memguard**，不是地板 2.5 太低。
本次走 user systemd、memguard 在位，**保持默认 2.5**，改用 §4.3 的加载后核对闸门兜底。

**本结论的边界**：只测了**单流**。PARALLEL=4 时四路大 prompt 同时 prepare 会叠加预留，
但按 24 B/字节算量级仍在几十 MiB；并发下的实测未做。

---

## 3. 三个硬约束（B 因此不是单变量）

### 3.1 POOL 整除 —— 不改就拒绝启动

`cap % (parts * chunk_rows) != 0 → error.BadPoolSize`

| chunk | 需 POOL % | 当前 POOL=1052672 |
|---|---|---|
| 2048 | 4096 | ✓ 余 0 |
| **4096** | **8192** | **✗ 余 4096 → 拒绝启动** |
| 8192 | 16384 | ✗ 余 4096 |

**候选 POOL = 1048576（2^20）**：对 2048/4096/8192/16384 全部整除 ✓，
且 = 2×WINDOW 正好满足下界，且是 2 的幂（满足 GRAPHS=1 对 pool 的要求）。
相对当前**减少** 4096 个 position（−0.39%），KV 内存略降，方向有利。

### 3.2 parts 断言 —— 两个常量必须同改

`model.zig` 与 `parts.zig` 各有一份 `chunk_rows`，`lanes.zig` 断言相等。只改一处会编译/运行期失败。

### 3.3 spill identity —— 现有 64 GiB tier 会失配

chunk_rows 进 tier identity ⇒ 旧块作废需重建。NVMe 上（head 1.1T 空闲），不吃 RAM，
但启动会变慢，且需在验收里确认 spill 能正常重建。

---

## 4. B1 方案（待用户批准后执行）

**范围：只试 4096，不试 8192。**

### 4.1 改动清单

| # | 对象 | 改动 | 可逆性 |
|---|---|---|---|
| 1 | **新增** `zig/patches/0068-prefill-chunk-4096.patch` | `model.zig` 与 `parts.zig` 的 `chunk_rows` 2048→4096 | 删除该文件即回滚 |
| 2 | `scripts/local.sh` | `POOL` 1052672 → 1048576 | 改回原值 |
| 3 | 实验期 env | ~~`MEMGUARD_GIB` 2.5 → 4.0~~ **撤回，保持 2.5**（见 §2.5 自我修正） | — |

**采用新增 patch 而非改旧 patch**：`model.zig` 那行在 0008 里是**上下文行**（上游已有），
直接改会破坏 patch 应用；Dockerfile 按 glob 顺序 `git apply --check && git apply`，
新增 0068 在最后应用，看到的是上游原值 2048，改成 4096 干净且符合「只增不改」纪律。
`growth_rows = 16*chunk_rows` 自动跟随，无需另改。

### 4.2 混淆变量声明（必须写进结论）

B1 同时动了 **chunk_rows** 与 **POOL**，**不是单变量 A/B**。
POOL 变化仅 −0.39% 且方向是省内存，但不得声称收益全部归因于 chunk。
spill tier 重建也是同期变化。

### 4.3 安全措施（对齐 `docs/performance/01` §0）

1. **走 user systemd**（`systemctl --user restart dsv41-tensorfold`），不手动 `docker run`，
   不绕过 `start.sh` 的 `memguard_start`。
2. **MEMGUARD_GIB 临时提到 4.0**：若实际 arena 增量超出测算上界，
   memguard 会在**加载阶段**杀掉容器，而不是等系统冻死。这是本次相对上次事故的关键改进。
3. **加载后先读日志里的 `arena … MiB used` 与 `Memory after the load`，核对内存账，再发任何流量。**
   若 arena 显著超出 7532 MiB 上界 → 立即停服回滚，不发请求。
4. 每步前后查 `/proc/meminfo`；不跑任何 kernel replay；不跑 ncu。
5. 构建预计 **3–8 分钟**（GitHub 可达 200；构建缓存 145 层 13.01 GB，
   `TF_REF` 不变 ⇒ clone 层命中缓存，只重跑 `COPY zig/patches` 之后的 zig 编译）。

### 4.4 回滚

删 `0068` patch → `POOL` 改回 1052672 → `MEMGUARD_GIB` 改回 2.5 →
`prepare.sh` 重建回原 PATCHES_HASH 镜像（原镜像 `exl3-tp2-8348338a8d90` **仍在本地**，可直接切回）→
`systemctl --user restart`。原镜像保留是本方案的安全底牌。

---

## 5. 预期收益：只说机制，不做比例连乘

R2048 时每专家 M = 2048×6/384 = 32；R4096 时 M = 64。
屋顶实测专家 bmm（权重恒 13.59 GB）：**M32 60.426 ms → M64 60.796 ms**，
即 M 翻倍而时间几乎不变（+0.6%）——这是「每 token 专家成本减半」的**机制依据**。

形状上，10755 token 从 `5×2048+515`（6 chunk）变成 `2×4096+2563`（3 chunk）。

但 E-DRAM-01 已判决真实内核只有 **123.1 GB/s = 屋顶的 50%**，且**实测占用率中位数仅 17%**。
chunk 翻倍会同时抬高占用率，收益可能大于也可能小于上述机制投影。
**因此不预告具体 tok/s，以 A/B 实测为准**（纪律：禁止比例连乘推收益）。

---

## 6. 边界与不确定性

1. 源码 ref **f77d5a1 ≠ 生产 pin fbbc9713**，分配公式可能有版本差异 ⇒ 本测算是**估计**，
   必须以启动日志实测 arena 复核。
2. engram / grouped-linear / prompt2d.Two 的 cap 线性项**未逐项枚举**，是上下界之间缺口的主因。
3. `slots=7`、`blas_ws≈32 MiB`、`ws.w≈42 MiB` 为推定值，未从运行时取实。
4. 本轮**未做任何构建、未改配置、未动服务**；服务全程健康。
5. 8192 档的「不安全」是**最坏情形判定**；若真实增量落在下界附近（3.94 GiB）理论上能装下，
   但余量仅 1.78 GiB，且无法在构建前证实 ⇒ **不做**。

---

## 7. 产物

- `arena_account.py` — 可复跑的测算脚本（含全部常量与公式来源注释）
- `arena-account.txt` — 本次运行输出
- 本文件