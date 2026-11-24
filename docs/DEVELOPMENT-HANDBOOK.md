# 开发手册（项目必读）

> 本文件是本项目的**开发方法论宪法**。任何对引擎、patch、配置、性能的改动，先读本文件再动手。
> 它记录的每一条都是从实际返工中提炼的，不是理论建议。
> 配套：`docs/DEPLOY-v075-NOTES.md`（部署坑与脚本底稿）、`docs/performance/`（性能台账）。

---

## 0. 总纲

**我方基线是已经过生产验证的最佳版本。社区不一定更好。**

引入社区改动的唯一目的是**在我方优化之上再吃到增量收益**。
因此：**没有可验证的增量时，基线就是最优解，不动它就是正确决策。**
「上游更新了所以我们该跟上」不是理由。

---

## 1. 方向：把社区特性摘到我方基线上，而不是把我方 patch 搬上游

| 错误方向 | 正确方向 |
|---|---|
| 以上游新版本为底，把我方 52 个 patch 往上搬 | **以我方基线为底**，把社区特性逐个摘进来 |
| 大爆炸换底，一次改动上万行 | 每次只动一个特性，通常几十行 |
| 总账持平时无法归因，只能到处找「丢了什么」 | 每一步都能独立归因 |
| 我方优势在搬运中可能被稀释 | 我方优势天然保持不动 |

**基线树的构造**（已验证可精确复现）：
```bash
cd <upstream-clone>
git worktree add -f --detach /tmp/tf-base <我方pin>
cd /tmp/tf-base
for f in <repo>/zig/patches/*.patch; do git apply --3way "$f"; done
# 期望：52/52 干净落地，零冲突。任何冲突都说明 pin 或 patch 集不对，先查明再动。
```

---

## 2. 吸星大法：每个社区特性的六步流程

**顺序不可颠倒，尤其不可跳过第 1、2 步直接构建运行。**

1. **读代码**。看这个 MR/commit 到底改了什么、**为什么**改、**怎么**实现的。
   读 commit message 的正文（作者通常写明了动机与实测数字），再读 diff 本体。
2. **判断对我方是否有益**。结合我方的真实工况（≤4 路并发、KV 分片开、内存红线、
   EAQ v1.1 混合包、我方 kit）。**很多社区改动修的是我方根本不存在的坑。**
   无益就明确跳过并写明理由，不要「先合了再说」。
3. **定吸收方式**。三选一：
   - **cherry-pick**：改动小、与我方结构兼容；
   - **重写**：上游实现依赖我方没有的结构，或我方有更优路径；
   - **启发**：不取它的代码，只取它的思路，做一个更适合我方的实现。
4. **同场 A/B**。见 §3。必须同场重测对照臂。
5. **确认收益且与预期相符**。收益方向对但量级差很多，说明机制理解有误，**回去查因，不要收下**。
6. **并入基线，抬高地板**。commit + 更新台账。然后看下一个，重复 1–6。

### 2.1 判断「是否有益」的实战样例

| 社区改动 | 它解决的问题 | 我方判定 |
|---|---|---|
| `share_check` | `TF_FILL_CHUNKS` 开着时独占 pass 也被切碎，8K 慢 2.2% | **跳过**：我方未设 `TF_FILL_CHUNKS`，pass 本就整段跑，无此坑 |
| `TF_DS_PASS_FROM` | 8192 以上用 4096 行 pass 才划算 | 需 4096 行 → arena 6144 MiB → **撞内存红线**，不能单独吸收 |
| `TF_DS_PROMPT_ALIAS` | prompt scratch 归并省内存 | 本身不提速，但**恰好腾出上一条所需的内存** |

**关键教训：候选之间存在耦合。** 上表后两条单看都会得出「没用」或「开不起」的错误结论，
合起来才是一条有明确预期收益的路径。**审候选时要看它们能不能互相解锁。**

### 2.2 「社区新增提交数」是假象，必须按内容核实

`git log <我方pin>..<上游tag>` 的数量**严重高估**真实增量：
同一改动常经由另一条分支线早已进入我方 pin，合并拓扑仍把它算成「新增」。

**正确做法：按符号/env 名在我方基线树里 grep，确认内容是否已在位。**
```bash
cd /tmp/tf-base && grep -rn "TF_DS_XXX" zig/src | head
```
实测教训：某次列出 2211 个「新增」提交，其中两个标着 `perf(dsv41)` 的核心开关
（`TF_DS_NO_COPIES`、`TF_DS_SOLO_OVERHEAD`）**我方基线早就有了**。
先 grep 30 秒就能知道，盲目 apply 会绕一大圈。

### 2.3 候选筛选的六道闸（先过闸，再读码）

一个提交标题再对路，先过这六道闸，**过半候选死在这里**：

| 闸 | 判据 | 本轮实例 |
|---|---|---|
| **后端** | Metal ≠ CUDA | `dd7c516a4`「每权重一窗口只读一次」只改 `kernels/metal/` |
| **家族** | Nemotron / lfm2 / glm ≠ dsv41 | `db7364e63` loads-in-flight 只改 `nemotron_*.cu` |
| **拓扑** | TP4 / four nodes ≠ 我方 TP2 | `c7aef7339` parity 专家切分明写 four nodes |
| **量化** | EXLR8 / NVFP4 ≠ 我方 exl3-2.9bpw | `51426b8e6` EXLR8 trellis |
| **规模** | >4 路 / 64 行轮 ≠ 我方 ≤4 路 | `06afb468e`、`c79c714ca`、`d27e3d422` |
| **可改性** | **当前树无源码 ≠ 上游无源码** | **E-SELF-02：cubin 的源码在另一条线的历史里，查全 ref 才找到** |

**查法**：`git diff-tree --no-commit-id --name-only -r <hash>` 看改的是哪些文件——
文件路径直接暴露后端与家族，比读标题可靠。
共享目录（如 `src/cuda/`）还须再确认**我方家族是否真的 import 它**：
`dc9d8bb6c` 改共享 `cuda/norms.zig`，但 dsv41 无该 import，等于不适用。

**第六道「可改性闸」的查法（自研尤其必过）**：
按 mangled 名反查我方家族真正调用哪个内核，再确认那个内核**有没有源码**：

```bash
grep -rn "_ZN" zig/src/families/dsv41/*.zig      # 我方真正的内核名
grep -rln "<kernel_symbol>" --include=*.cu .      # 0 命中 ⇒ 可能是预编译件
```

**我方 exl3 专家/线性路径以 kit 预编译 cubin 分发**（`kit/cubins/*.cubin`，kit 不发 .cu），
但**源码在上游公开**——`c99083245` 的 `src/tensorfold/cuda/exl3/experts_grouped.cuh`（2283 行）
+ `experts_cb2.cu`（14 行，自包含）。**「当前 worktree 里没有」不等于「拿不到」。**

**判「无源码」前必须查全**（E-SELF-02 就是漏了这步而误判）：

```bash
git log --all --oneline --name-only --diff-filter=AM -- "*<关键文件名>*"   # 全 ref 全历史
git ls-tree -r --name-only <候选commit> | grep -i <关键词>                # 该版路径
```

注意源码可能在**另一条线**（Python 线 `src/tensorfold/` vs Zig 线 `zig/`），
且同名文件的**不同版本模板参数个数不同**（6 参数 `grouped_kernel` vs 8 参数 `grouped_rows_kernel`），
**必须用签名与实例化配置对上我方 cubin 才算找到对的那份**。

**过闸后剩下的，若代码不适用但机制对路，归入「启发自研」而非丢弃**（见 §2 第三步）。

### 2.3.1 启发自研前必做「机制同构核查」

不是问「我方有没有这个 commit」，而是问
**「我方 kernel 有没有这个机制、是什么形式、孰强孰弱」**。

**顺序不可颠倒：① 先测量定位靶子大小 → ② 再读码核实机制是否已存在 → ③ 才谈实现。**

实测教训（E-SELF-01）：打算把 Nemotron 的「让 load 保持在飞行中」移植到 dsv41。
- 第①步 profiling 发现：该改法的靶子 router/topk 在我方 decode 里只占 **m_rt 629.9 µs / 轮 ≈ 整轮 2%**，
  而 **moe 占 12909 µs（stretch 口径 78.2%）**。**靶子选错了。**
- 第②步读码发现：我方 `lane_gemv.cu` 是 `gemv_kernel<64,64,8>`，
  `cp.async.cg.shared.global` 16B，主循环前灌 7 级、循环内提前 7 级发射；
  `experts.cuh` / `qmm_group.cu`（8 级）同样。**机制早已存在，且是比上游寄存器 staging 更强的多级软件流水。**

**两步都在写任何代码之前完成，避免了一次实现已有功能的无效开发。**

### 2.3.2 decode 分段 profiling 的正确取法

`RANK1_PROFILE=1` + `TF_DS_CLOCK_MARKS=1` → 跑探针 → **停服**（profile 在退出时才打印）。
- **容器会被停服流程重建，`docker logs` 拿不到**。去 worker 的
  `~/.cache/dsv41-tensorfold/logs/*rank1.log.gz` **归档**里 `zgrep round_clock_us`。
- `CLOCK_MARKS=1` 在 graph 内也打点，**打断 PDL 链、绝对值偏大**（实测 N=1 58.621 vs 基线 59.1）。
  **只用相对占比，绝对值不得当基线。**
- stretch 总段（`moe`/`attn`/…）与细标记（`a_*`/`sh_*`）是两套口径，**不可相加**。
- `m_gu`/`m_x` 计数为 0 属正常：那是 TP4 2D MoE 的标记。

### 2.4 上游的「无收益」结论不可跨配置外推

上游 commit body 里的实测结论绑定**它自己的硬件与拓扑**。
例：`28dbfe128` 写 "the settings swept on real one-stream decoding move nothing past noise"，
但那套 L2 prefetch budget 与配速是在 **TP4** 上调的，**每 rank 权重字节数只有我方 TP2 的一半**。
差异落在决定性维度上时，其零结论对我方无效，须自行重扫。

反之亦然：**上游的正收益同样不可直接外推**（参见 B 步 `TF_DS_PASS_FROM` 的 +5.6% 实为
hybrid vs 4096-everywhere，不是我方起点）。**两个方向都要按我方真实起点重算。**

---

## 3. 测量纪律（最重要，违反它会让全部结论作废）

### 3.1 任何 A/B 必须同场重测对照臂

**禁止复用历史基线数值做单次增量判决。**

本机同日状态漂移可达 **1%**，与待测增量同量级。
实际事故：3b 端口后用今日数据对比历史基线，得出「−1% 回退」，
据此追查「丢了哪个 patch」耗费大量工时；同场重测基线后发现
**3b 实际是 +0.11% / +0.36%，根本没有回退**。

- 历史基线**只能**用于长期趋势观察。
- 每个 A/B 批次内，对照臂与实验臂必须**同一时段、同一探针、同一重启流程**测出。
- 报告里必须写明对照臂是本次实测还是历史值。

### 3.2 口径

- prefill 一律用 **`未缓存输入token / server.prefill_seconds`**，不用端到端时间。
- decode 取 `decode_tok_s`，多 rep 时**除首次外取中位**（首次含预热）。
- 并发看**聚合吞吐**，同时记录每流值。
- 记录 `tok/轮` 与 `接受率`：decode = tok/轮 × 轮/秒，**两者必须分开看**才能归因。
  实例：`SOLO_OVERHEAD=1` 使 tok/轮 2.0157→2.0317 但接受率 0.5646→0.5417，净亏 2.22%。
  只看吞吐会以为是噪声，看分解才知道是「草稿更深但接受更少」。

### 3.3 禁止比例连乘推收益

不得用「A 比 B 快 x%，B 比 C 快 y%，所以 A 比 C 快 (1+x)(1+y)」代替实测。
每一格都要实测。2×2 表缺哪一格就补测哪一格。

### 3.4 探针选择偏差：必须补对抗性负载

**一个开关在合成探针上为正，不等于它在真实负载上为正。**

凡是改动**投机 / 复制 / 草稿机制**的开关，除了常规探针，**必须再跑一个最大化该机制贡献的
对抗性负载**，两个场景都过才可收下。

实际事故（E-ABS-02）：`TF_DS_NO_COPIES=1` 关闭 suffix-lookup 复制窗口。
- 常规探针（47 token 短 prompt → 生成 512 token，几乎无重复）：交替 A/B/A/B **四次配对全正**，
  N=1 +0.30%/+0.35%，N=4 +0.74%/+0.39%，看起来是真实小收益。
- 对抗探针（长重复文本 + 要求逐字复述）：**N=1 −4.87% / −4.46%**，
  tok/轮 从 3.5068 掉到 3.3247 —— 复制窗口本来在实打实产 token。

**常规短探针恰好是该开关的最有利情况。** 真实 agent 负载（工具输出回放、文件引用、
代码编辑、reasoning 重放）充满重复，对抗场景才具代表性。若只看常规探针，
就会把 −4.7% 的真实退化当成 +0.6% 的收益锁进基线。

**推论**：收益方向随负载翻转的开关，不得设为全局默认。

### 3.5 小效应必须交替多轮

待测效应 ≲1% 时，单次 A→B 抗不住漂移。用 **A/B/A/B 交替**，
看**每一对配对差是否同向**，并比较效应量与同配置内散布的比值。
符号一致性比单次幅度更有说服力。

---

## 4. 内存红线与操作纪律

- **生产 head 多路最小空闲 4.8 GiB 是硬红线。** 空载单流的富余不代表生产富余。
- 任何构建前：**先停服**，断言 `MemAvailable > 100 GiB`。
- **绝不并行跑两个引擎实例。**
- 不得下调 `TENSORFOLD_MEMORY_RESERVE_GIB`。
- **arena 由编译期常量定容**（`model.zig` 的 `prompt.Chunk.init(..., chunk_rows, ...)`）。
  `TF_DS_PASS_ROWS` 只限制实际 pass 长度，**改不动 arena**。要省 arena 必须改编译期常量。
- 用户不在场时爆内存需人工断电、可能十几小时无法恢复。**宁可少测一轮，不可越线。**

### 4.1 停服/恢复标准流程

```bash
systemctl --user stop dsv41-tensorfold.service
sleep 8
# 改配置
systemctl --user reset-failed dsv41-tensorfold.service   # StartLimitBurst=3/30min
systemctl --user start dsv41-tensorfold.service
# 轮询 /health 至 200；就绪约 50–95 s（视镜像而定）
```

---

## 5. 端口与合并的工程陷阱

1. **必须 `git apply --3way`**，直接 `git apply` 会大面积失败。
2. **逐 patch 解完冲突并 `git add -A` 后，才能落下一个。**
   否则报 `does not exist in index`，后续 patch 整体失败、新文件不生成。
   （此坑本人踩过两次，写在这里就是为了不再有第三次。）
3. **改枚举编号要改全部三处**：枚举定义、对应模块的 `kind` 常量、
   **字节→枚举的解码 switch**。漏第三处 → 对端 `BadFrame` → 本端 `PairBroken` → 进程退出。
4. **`SafeAllocator leaked` 是错误后的清理噪声，是果不是因。** 排错时先找首个真实错误。
5. **上游可能拆分了我方单一常量的语义**。例：我方 `chunk_rows=2048` 被上游拆成
   `chunk_rows=4096`（pass 大小）+ `resume_rows=2048`（保留态网格）。
   我方按块粒度做保留态判定的地方，对应的是 `resume_rows`，断言要写对。
6. **上游重命名了我方 env**：我方 `TF_DS_POOL` ↔ 上游 `TF_DS_WINDOW`；
   我方 `TF_DS_WINDOW` ↔ 上游 `--context`。端口时要做映射，不能照抄名字。
7. **自包含新文件可单独提取**，不必端口整个大 patch。
   例：KV 分片依赖的 `rank_wire.zig`/`fingerprint.zig` 由 8699 行的 spill patch 新建，
   但两者自包含，单独提取即可，冲突从 71 块降到 17 块。
8. 解冲突原则：**上游演进处取上游，我方独有功能加回去**；
   theirs 混入的未端口依赖只挑需要的行；上游移动了符号则改挂到上游模块。

---

## 6. 如何确认一个特性「真的生效」

**不能只看「服务起来了」。必须拿到直接证据。**

| 证据类型 | 做法 |
|---|---|
| env 真进了容器 | `docker exec <容器> env \| grep TF_DS_XXX` |
| 启动日志实证 | 抓 `pool ... positions` / `arena ... MiB used` / 特性自己的 info 行 |
| 符号真在镜像里 | `strings <二进制> \| grep <kernel符号>` |
| 没有静默回退 | 抓该特性的 warn 行（例：ring kernel 缺失会 warn 并退回 NCCL） |
| 内存数对得上 | 分片开：532 MiB/rank；关：893 MiB 复制 |

**注意日志归档时机**：`*.log.gz` 是 stop 时归档的**上一次**运行。
看当前运行要用 `journalctl --user -u dsv41-tensorfold.service`，否则会拿旧数据当新数据。

**遥测缺口 ≠ 功能缺陷**：`/health` 某字段为 null 可能只是遥测链路没打通，
功能本身已由日志与内存数证实生效。两者要分开陈述，不可互相推断。

### 6.1 env 旋钮必须先过我方启动脚本的白名单

**上游能读的 env，我方部署不一定送得进容器。** `start.sh:291` 只直通
`^(TENSORFOLD|TF_DS|TF_SPILL|TF_FILL)_` 前缀，且另有排除表与 port 映射表。

两个实测陷阱：
- **前缀不匹配**：`TF_EXL3_L2_DELAY_NS` / `TF_EXL3_L2_RATE_GBPS` 属 `TF_EXL3_`，
  **一个都进不去**。不查就测，等于连测两臂空开关还得出一堆「噪声内」的假数据。
- **被 port 映射接管**：`TF_DS_L2_PREFETCH` 在排除表里，由 `PREFETCH` 经
  `start.sh:339` 注入。**直接 `export TF_DS_L2_PREFETCH=0` 无效，必须改 `PREFETCH=0`。**

**纪律：每吸收一个带旋钮的特性，逐个旋钮在容器里 `env | grep` 核验取值，
而不是只核验其中一个。** 扫描类实验尤其致命——某一臂没生效会伪装成「该参数不敏感」。

---

## 7. 记录与归档纪律

- **Git 仓库是本项目唯一标准。** Spark 上的项目目录不是 git 仓库，改动须 scp 同步回来。
- **实验只增不改**：结论被推翻时，**原文保留 + 追加「更正批注」**，不删原文。
  作废的推算要显式写明「该推算作废」及原因。
- **写前先读**：任何要写入的目录先 Read/Grep，避免覆盖既有权威产物。
- **review 后 commit**；commit 前扫描敏感串，hits 必须为 0。
- **>1 min 的构建/实测先报预计时长。**
- 每个实验建 `runs/forge-<YYYYMMDD>-<主题>/`，含 `REPORT.md` + `raw/` 原始数据。
- 长等待要给可见进度，不要静默数分钟。

### 7.1 判决台账模板

每个候选特性都要留下这样一行：

| 字段 | 内容 |
|---|---|
| 社区坐标 | commit hash + 标题 |
| 做什么/为什么/怎么做 | 读代码后的三句话 |
| 对我方是否有益 | 是/否 + **理由**（结合我方工况） |
| 吸收方式 | cherry-pick / 重写 / 启发 / 跳过 |
| 同场 A/B | 对照臂实测值、实验臂实测值、差值 |
| 预期 vs 实际 | 是否相符；不符则查因 |
| 判决 | 收下并入基线 / 否决 / 待复测 / 条件候选 |

---

### 7.2 断言「某问题仍未解决」前，先查实验索引

**必须先查 [`docs/performance/03-实验与证据索引.md`](<performance/03-实验与证据索引.md>)，
而不是只读最早提出该问题的那份报告。**

提出问题的报告写在前，解决它的报告写在后。只读前者会把**已解决**的问题当成开放的，
并据此做出错误的方向判决。

实际事故（E-SELF-01 §6 更正批注）：读 E-ROOF-01 见「2.00× 缺口两假设未分离」，
据此写下「仍开放」并判自研方向无空间。而 **E-DRAM-01 r2 早已分离**：
H1 否证、H2 成立，机制是**低占用延迟受限**（Occupancy 中位 17%）。
漏读索引直接导致**判漏了「提高占用率」这个真实杠杆**。

**推论**：任何「未做 / 未分离 / 未验证」的断言，都要在索引里核一遍再写。
状态与数值的唯一口径是 [`docs/CURRENT-BASELINE.md`](<CURRENT-BASELINE.md>)。

### 7.3 「有机制」≠「机制够用」

判断某优化是否已在位，不能只问「有没有这个机制」，还要问「**它的量够不够**」。

实例：我方专家 kernel 已有 cp.async **7–8 级**软件流水（形式强于上游的寄存器 staging），
但**实测 Occupancy 中位仅 17%**（`down` 255 regs/thread、128 threads/block ⇒ 8/48 warps），
**驻留 warp 太少 ⇒ 总在飞请求数仍不足以盖住延迟**。
**「有流水」不等于「在飞请求够多」。** 只看到前者就判「无实施空间」是错的。

> **【E-SELF-02 更正】** 上段「7–8 级 cp.async」读的是 `lane_gemv.cu` / `experts.cuh` /
> `qmm_group.cu`，**这些属于 Nemotron 家族，我方 dsv41 不调用**。我方 exl3 内核无源码，
> 流水结构未知。255 regs / 16.7% 那部分**已从二进制核实为真**
> （`tf_exl3x::grouped_rows_kernel<2,8,4,1,2,16,2,false>`，REG:255、STACK:0）。

### 7.4 静态占用率必须从二进制读，且要列全同族变体

**不要沿用文档里的 regs 数字**，用工具实测：

```bash
bash tools/kernel-resusage.sh            # kit cubin(cuobjdump) + 引擎 kernel(nvcc -Xptxas -v)
```

**判据**：静态上界 = ⌊65536 / (regs × threads)⌋ blocks × (threads/32) warps / 48。
shared 与寄存器分别算，**取更紧的那个**才是真限制因子。

**两条本轮踩过的坑**：

1. **只筛一个实例化就下结论**。我曾只筛 decode 的 `grouped_cp_kernel`（128 regs），
   就宣称「255 不存在、E-DRAM-01 错了」——**错**。255 属于 prefill 的 `grouped_rows_kernel`。
   **同族变体必须全量列出再判决**（`raw/kit-exl3x-named-resusage.txt` 有 86 个）。
2. **把「当前树没有」当成「拿不到」**。cubin 无源码是真的，但**上游公开源码存在**，
   在另一条线的历史里（`c99083245`）。**判「不可实施」前必须查全上游所有 ref 与历史路径**
   （见 §2.3 第六道闸的查法）。这是本轮最贵的一个误判。
3. **静态占用率低 ≠ 抬高它有收益**。E-SELF-02 路 A 实测：两个变体都把 16.7% 抬到 25%，
   却分别 −0.92% / −2.38%。**占用率只是中间量，必须落到吞吐上判**；
   且要看清换占用率付的是什么代价（复用？字节？工作粒度？），三者代价不同。

### 7.5 构建成功判定：`grep BUILD_OK` 在 `set -x` 日志上是假阳性

`build-alias.sh` 带 `set -x`，日志会把 **`echo BUILD_OK` 这条命令原文**也记进去。
于是 `grep -c BUILD_OK` 在构建**尚未结束**时就返回 1。本轮据此提前拷贝二进制，
做出了一个**不含改动的镜像**，白跑一轮冒烟。

**正确判据**（任一）：

```bash
tail -1 ~/build-x.log            # 末行必须是 BUILD_OK 本身
# 或等后台任务真正结束再取产物
```

**并且产物要验内容，不验存在**：改动的日志串是否真在二进制里——

```bash
strings ~/build-x-out/bin/tf-dsv41-lanes | grep -c "<新日志的关键词>"
docker run --rm --entrypoint bash <派生镜像> -c "strings /opt/tensorfold/bin/tf-dsv41-lanes | grep -c '<关键词>'"
```

**运行态还要看引擎自报**：env 到达容器 ≠ 改动生效。本轮靠引擎自己打的
`grouped_rows: … nt 4 pf 2` 才确认变体真被取用；只有 env 而无该行时，就是镜像里二进制不对。

---

## 8. 已验证做不通的路（勿重试）

- 驻留态 ncu kernel replay（爆内存，需人工断电）
- `TF_DS_IDX_SPLIT=1`（−1.9%，短 chunk 的 all-gather 开销大于省下的计算）
- `TF_DS_SOLO_OVERHEAD=1`（−2.22%，草稿深但接受率降）
- `TF_DS_NO_COPIES=1` 作全局默认（常规探针 +0.3~0.6%，但**重复文本单流 −4.7%**；见 §3.4）
- **L2 prefetch 参数扫描**（`TF_DS_L2_{ATTN,MOE,NEXT,STRETCH_END}_MB` 加倍/减半、`TF_EXL3_L2_RATE_GBPS=0`）：
  八臂实测全在噪声内，加倍对 N=4 轻微有害（−0.4~0.5%）。上游 TP4 的零结论**在我方 TP2 复现**。
  注意 **`TF_DS_L2_PREFETCH` 本身必须保持 1**：关掉亏 **6.3% 单流 / 3.8% 四路**（这是开关，不是参数）。
- `TF_DS_PASS_FROM` / 4096 行 pass 对我方短 prompt：8192 以下**构造上零收益**；
  我方 10752 token 探针实测 4096 vs 2048 仅 +0.05%；且 arena 需多 2272 MiB，
  alias 只补回 1558 MiB，**缺口 714 MiB**撞红线。上游 +5% 只在 128K 量级且其自身散布达 4%。
- `ROUND_ROWS` 在 ≤4 流下无效；`ROUND_ROWS=48` 更慢
- **提高 exl3 专家内核静态占用率（改 tile 形状）**（E-SELF-02 路 A，**已实测**）：
  cubin 自带的两个低寄存器变体都把静态占用率 16.7%→25%，**但都掉吞吐**——
  `TF_DS_EXPERT_G=1` **−0.92%**（4/4 配对全负，成员行 32→16 使权重 tile 复用减半、字节放大）；
  `TF_DS_EXPERT_NT=4`（pf=2,g=2）**−2.38%**（2/2 全负，复用与字节不变，亏在每 program 工作粒度/ILP）。
  **「静态占用率低」是事实，但不等于「抬高它就有收益」。** 源码改动已回滚，勿重试这两个形状。
  路 B（同形状加 `__launch_bounds__` minBlocks）未做，有实据的 spill 风险。
- ~~降 exl3 专家内核寄存器压力~~ **【此条目曾误判，见上条】** E-SELF-02 一度判「无源码不可实施」，
  **该判决错误**：源码公开在上游 `c99083245` 的 `src/tensorfold/cuda/exl3/experts_grouped.cuh`。
- chunk 8192（arena +10.20 GiB）；chunk 4096 未配 alias 时 arena 6144 MiB、avail 跌至 5G
- 权重/KV 卸载 SSD
- 整体换 tp2 分支（50/52 patch 失败）
- 裸 v0.7.5 加载我方 checkpoint（mixed pack 0 命中）
- Spark 侧 git fetch GitHub（TLS GnuTLS -110）
- 改网口 MTU 1024→4096（带宽不变、端到端 −0.20%）
- service 运行时跑 `DRY_RUN=1 ./start.sh`（持 start.lock）

---

## 9. 环境速查

- head `ztxjzh@192.168.1.118`、worker `ztxjzh@192.168.100.11`
  （**主机名被代理劫持，必须用 IP**；Mac 直连 worker 超时，须经 head 跳）
- API `http://127.0.0.1:8888`，key 在 `~/.local/state/dsv41-tensorfold/api-key`
- 探针：`/tmp/decode_probe.py <label> [max_tokens] [temp] [reps]`、
  `/tmp/decode_conc.py <N> [...]`、`/tmp/cold_prompt.py <out.json>`
  （`/tmp` 重启清空；脚本无执行位时用 `bash` 调用）
- Mac 类型检查：`zig build test-cuda-host`，剩余失败为 macOS 固有
  （`link.zig` 的 Linux 专有 `c.SO.BUSY_POLL` + 若干 benchmark 计时测试）
- 生产基线权威数值见 `docs/performance/BASELINE.md`，
  **但据 §3.1，它不可直接用于单次 A/B 判决。**