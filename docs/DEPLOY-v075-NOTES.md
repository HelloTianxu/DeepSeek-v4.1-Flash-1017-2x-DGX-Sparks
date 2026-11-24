# v0.7.5 部署知识底稿（供一键脚本构造）

> **⚠️ 路线声明：本文所属的「换 v0.7.5 底座」路线已搁置。**
> 生产跑的是 `base-alias`（旧 pin `fbbc9713` + 53 patch），**不是** `v075-3b`。
> 当前基线口径见 [CURRENT-BASELINE.md](<CURRENT-BASELINE.md>)；
> 一键脚本已由 `tools/deploy-baseline.sh` + `tools/verify-baseline.sh` + `tools/check-boot.sh` 实现（走基线路线）。
> **本文的构建手法、派生镜像 Dockerfile、排错案例（J5 等）仍可复用**，路线结论不可当当前状态。原文保留不改。

**目的**：把「v0.7.5 引擎 + 我方承重 patch」这条路上踩过的每个坑，记成**可脚本化**的形式。
每条按 `症状 → 根因 → 解法 → 验证` 组织，附精确命令与判据。

**状态**：**3a 全部完成并已验证**。RDMA 与 graph-cap 两个阻塞项均已修复实测通过，
公平性能对比已完成（见 I 节）。生产已恢复原配置。

---

## A. 环境前置事实（脚本必须先断言）

| 项 | 值 | 断言方式 |
|---|---|---|
| 引擎仓库 | `https://github.com/bertholomus/TensorFold.git` | — |
| **v0.7.5 引擎 pin** | `e100ee8b1d806a9f516d7a01cbcbf06ae98d77e1` | kit 的 `ENGINE_COMMIT` 文件 |
| **配套 kit pin** | `fdde3285fd761add38e6d4f703f5402936f37220` | kit 仓库 main |
| 我方旧 pin（回滚用） | `fbbc97132cd004237d9887b8d3185896efaedc5c` | — |
| **Zig 版本** | **恰好 0.17.0** | `build.zig:25` 硬 `@compileError`；0.17.1 **不行** |
| 构建目标 | `-Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=ReleaseFast install native` | GB10 = SM 12.1 |
| 产物 | `/opt/tensorfold/native/bin/tensorfold-native`、`/opt/tensorfold/bin/tf-dsv41-lanes` | `test -x` |
| head / worker | `ztxjzh@192.168.1.118` / `ztxjzh@192.168.100.11` | **主机名被本机代理劫持到 fake-IP，必须用 IP** |
| 服务 | user systemd `dsv41-tensorfold.service` | `systemctl --user` |
| 就绪时间 | **约 48 s** | 轮询 `/health` == 200 |
| 重启限流 | `StartLimitBurst=3 / 30min` | 超限先 `systemctl --user reset-failed` |
| MemTotal | 121.63 GiB | — |
| 生产可用基线 | **4.8 GiB**（多路），空载 7.8–8.2 | **判定用 4.8 而非空载读数** |

---

## B. 已知坑（按发现顺序）

### B1 · Spark 无法从 GitHub 拉代码

- **症状**：`git fetch` 报 `curl 28 GnuTLS recv error -110`
- **根因**：Spark 侧到 GitHub 的 TLS 通路不通
- **解法**：**Mac 克隆 → tar over ssh → Spark**，绕过 Spark 的网络
- **副作用坑**：macOS `tar` 产生 AppleDouble `._*` 文件（795 vs 394）
  → **必须 `find <dir> -name "._*" -delete`**
- **验证**：文件数与源端一致；kit 的 `MANIFEST` 383 条 0 不符

### B2 · 裸 v0.7.5 加载不了我方 checkpoint

- **症状**：（预判，未实际触发）27 个 BF16 linear 被当 EXL3 trellis
- **根因**：v0.7.5 中 `mixed pack` / `plain_entries` / `plain_rows` **全 0 命中**；
  上游只服务纯 EXL3 2.5bpw，我方是 **EAQ v1.1 混合包 + ablit**
- **解法**：必须带 `0024-mixed-pack` + `0025-widen-bf16` + `0026-plain-rows`
- **验证**：日志出现 **`info: mixed pack: 27 float linears`**

### B3 · `TF_DS_POOL` / `TF_DS_WINDOW` 语义反转 ⚠

- **症状**：两 rank 池不对称（rank0 得 524288、rank1 得 1052672）
- **根因**：
  - 我方 `start.sh:286`：`TF_DS_POOL=$POOL`、`TF_DS_WINDOW=$CONTEXT`
  - **v0.7.5 完全不读 `TF_DS_POOL`**（0 命中），其 `pool = TF_DS_WINDOW 或 --context`
  - **我方 WINDOW = 单请求窗口；上游 WINDOW = 共享池，含义相反**
- **解法**：跑 v0.7.5 时 **`POOL` 留空**，两 rank 同取 `--context`
- **验证**：`DRY_RUN=1 bash ./start.sh` 检查 rank1 `--pool` == rank0 `--context`

> **【3b 更正批注】** 上述解法只适用于 **3a/r2（未端口 KV 分片）**。
> 3b 起 `native.zig` 已改为 `.pool = pool_mod.resolve(o.context, getenv("TF_DS_POOL") orelse getenv("TF_DS_WINDOW"))`，
> **`TF_DS_POOL` 重新生效且优先**，我方 `start.sh` / `port.sh` 的原有导出可直接沿用，**`POOL` 不再需要留空**。
> 映射关系（脚本必须理解）：
>
> | 概念 | 我方名 | 上游 v0.7.5 原名 | 3b 后 |
> |---|---|---|---|
> | 共享池（≥context） | `TF_DS_POOL` | `TF_DS_WINDOW` | 两者都读，POOL 优先 |
> | 单请求窗口 | `TF_DS_WINDOW` | `--context` | `--context` |
> | 单流上限（代码符号） | `request_cap` | `ctx_cap` | 统一用 `ctx_cap` |

### B4 · `--graph-cap` 上游不存在

- **症状**：GRAPHS=1 时 rank1 拒启（未知 flag）
- **根因**：`--graph-cap` 是我方 `0033` 加的，上游无
- **临时解**：`GRAPHS=0`（但**牺牲性能**，且无 graph 加速）
- **正解**：端口 `0033` —— **已完成并验证**，3 个冲突（round/native/lanes_main），新建 `graph_cache.zig`
- **验证**：`DRY_RUN=1` 出现 `--graph-cap 128 --graphs 1`，启动后推理正确
- **⚠ 内存风险**：无 0033 时 graph cache **无界增长**（设备内存 + ~1.6× 主机内存），
  跑数天后 memguard 杀 rank0。**长期运行必须有 0033**

### B5 · RDMA 推理失败 `error.Failed` ← 本轮核心发现

- **症状**：加载正常、`/health` 200，但推理 0.15 s 即 `{"error":{"message":"Failed"}}`；
  rank1 日志 `{"follow_refused": "Failed"}`；NCCL（RDMA=0）下**完全正常**
- **定位路径**：
  1. `grep error.Failed` → `rdma.zig:402/433` 的 `gather()`
  2. 读注释 → 「a watchdog lost a peer」，`failure()` 返回 `why` 但**未打日志**
  3. 失败源全是 verbs 级：`ibv_post_send/poll_cq/post_recv failed`、WC status
  4. **对比 pin 与 v0.7.5 的 rdma.zig diff**（45 增 87 删）→ 找到关键行
- **根因**：**v0.7.5 硬编码 `path_mtu = mtu_4096`（5）**，
  而我方 pin 是 `pathMtu(local, remote)` **协商实际 active MTU**。
  Spark 两个 RoCE 口实测：`max_mtu 4096 (5)` 但 **`active_mtu 1024 (3)`**
  ⇒ QP 按超过链路能力的 MTU 配置 → verbs 失败
- **验证根因的命令**：
  ```bash
  for d in rocep1s0f1 roceP2p1s0f1; do ibv_devinfo -d $d | grep -E "active_mtu|max_mtu"; done
  ```
- **解法（选此）**：**把 MTU 协商回port 到 v0.7.5**，7 处改动：
  1. `Error` 集合加 `BadMtu`
  2. `Info` 加 `active_mtu: [max_devices]c_int`（两 rank 同二进制，布局一致）
  3. `Nic` 加 `active_mtu: c_int`
  4. 加 `pathMtu()` 函数（照抄 pin：范围校验 + `@min`）
  5. `openNic` 读 `r.v.port(ctx,1).active_mtu`
  6. `info()` 填 `out.active_mtu[d]`
  7. `connect()` 先校验全部对端 MTU，再 `path_mtu = try pathMtu(n.active_mtu, ri.active_mtu[d])`
- **未选的方案**：改网口 MTU 到 4096。**理由**：重启会丢；且我方实测过
  「RoCE MTU 1024→4096 带宽不变、端到端 −0.20%」，无收益
- **✅ 已验证修复**：回port 后 RDMA=1 + GRAPHS=1 全开，推理正确（回答 "7"，reasoning 正常）
- **排除项**：v0.7.5 **不读** `TF_RDMA_IDLE_S` / `TF_RDMA_NAP_US`（0 命中），
  我方这两个变量被忽略，**不是它们的问题**；`0010`（idle-nap）是省电优化非正确性修复，
  `0012` 对 rdma 的改动**只是给 0010 加 Linux prctl 守卫**，两者都不是根因

### B6 · `rm -rf` 失败导致 `&&` 短路，源码根本没展开

- **症状**：构建启动后报 `rdma.zig: 没有那个文件或目录`
- **根因**：容器内构建产生的 `.zig-cache` 是 **root 属主**，用户 `rm -rf` 删不掉返回非零，
  `rm -rf X && mkdir X && tar xzf ...` 的 **`&&` 短路**，tar 从未执行
- **解法**：**每轮用全新目录**（`build-v075-r2`、`-r3`…），不复用、不删除
- **教训（写进脚本）**：跨容器构建目录**不要 `rm -rf` 后 `&&` 串联**；
  要么用新目录，要么 `rm -rf X || true` 再独立执行后续

### B7 · Mac 主机测试的固有失败（非我方引入，勿误判）

- `link.zig:23` 的 `c.SO.BUSY_POLL` —— **Linux 专有**，macOS 无此常量
- 3 个 benchmark 信号/计时测试（`benchmark_stop_test` ×2、`benchmark_stream_test` ×1）
- **判据**：`git diff --name-only | grep link.zig` 为空 ⇒ 非本次改动引入
- **正常基线**：**426/433 通过**（`zig build test-cuda-host`）。低于此数才是真问题

### B8 · 三方合并时 theirs 侧会混入未端口的依赖

- **症状**：`0026` 的 model.zig hunk 带来 `m.kvs` / `shard.Shard.init` / `Caches.initShard`；
  `0033` 的 round.zig hunk 带来 `shard_ring` / lanes_main 带来 `spill_tier`
- **根因**：我方 patch 是在**已含 KV 分片/spill 的树**上做的，context 行携带了那些代码
- **解法**：**只取本 patch 真正需要的那几行**，丢弃混入的 3b 依赖
  - `0026` → 只取 `m.eng.plain_rows = o.plain_rows;`
  - `0033` → 只取 `const graph_cache = @import("graph_cache.zig");`
- **验证**：`grep -rn "shard_ring\|spill_tier\|spillFlag" <改动文件>` 应为空

---

## C. 冲突解决原则（可复用到后续 patch）

**不是二选一，而是「上游演进处取上游，我方独有功能加回去」。**

| 场景 | 处理 |
|---|---|
| 两边是**不同函数**撞在同一位置 | **并存**（如 `procText` 与 parked 三件套） |
| 上游**演进了逻辑** | **取上游**，只把我方 instrumentation 加回（如 cuda.zig 保留 `learn_bytes`/`peer_room`/`free_streams`，加回 parked 日志） |
| 上游**移动了符号** | 改限定名（`procText` → `budget.procText`；`Pool` → `budget.Pool`） |
| 上游**已有等价测试** | **删除我方重复品**（`chipClass` 测试已在 `cuda_device.zig`） |
| 上游**已有等价功能** | **整个 patch 丢弃**（`0008` ↔ 原生 `--pool`/`--context`） |
| theirs 混入**未端口依赖** | 只取需要行（见 B8） |

---

## D. 当前最小 patch 集

| Patch | 状态 | 备注 |
|---|:--:|---|
| `0011-cuda-budget-parked-pages` | ✅ | 内存安全 |
| `0024-mixed-pack` | ✅ | **三方合并全干净** |
| `0025-widen-bf16` | ✅ | **全干净** |
| `0026-plain-rows` | ✅ | 2 文件冲突已解 |
| `0033-graph-cap` | ✅ | 3 冲突已解，新建 `graph_cache.zig` |
| **RDMA MTU 协商** | ✅ | **非原 patch，是从 pin 回port 的修复**（见 B5） |
| ~~`0008-pool-window`~~ | ❌ | 上游原生等价 |
| ~~`0012-dsv41-base-fixes`~~ | ❌ | 3a 不需要 |

---

## E. 构建与部署流程（脚本骨架）

```bash
# 1. Mac：取源码 + 打补丁
git worktree add /tmp/tf-v075 e100ee8b1d806a9f516d7a01cbcbf06ae98d77e1
cd /tmp/tf-v075 && git apply --3way <承重patches>   # 直接 apply 会失败，必须 --3way
# 手工解冲突（见 C 节原则）

# 2. Mac：类型检查（廉价，先做）
/tmp/zig0170/zig-aarch64-macos-0.17.0/zig build test-cuda-host   # 期望 426/433

# 3. 传输（每轮新目录，见 B6）
tar czf /tmp/src.tar.gz --exclude=.git --exclude=.zig-cache --exclude=zig-out .
scp /tmp/src.tar.gz head:/tmp/
ssh head 'mkdir -p ~/build-rN && tar xzf /tmp/src.tar.gz -C ~/build-rN;
          find ~/build-rN -name "._*" -delete'

# 4. 停服腾内存（必须！生产占 113 GiB）
ssh head 'systemctl --user stop dsv41-tensorfold.service && sleep 10 && free -g'
# 断言 available > 100 GiB

# 5. 构建（内存硬上限，绝不并行两引擎）
docker run --rm --memory=40g --memory-swap=40g --entrypoint bash \
  -v ~/build-rN:/tensorfold -v ~/build-rN-out:/opt/tensorfold \
  tensorfold-dsv41:exl3-tp2-8348338a8d90 -c \
  "cd /tensorfold && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 \
   -Doptimize=ReleaseFast --prefix /opt/tensorfold -j16 install native && echo BUILD_OK"

# 6. 派生镜像（两节点都要！rank1 在 worker 上跑）
#    Dockerfile: FROM <生产镜像> + COPY 两个二进制
ssh head 'cd ~/img && docker build -t tensorfold-dsv41:v075-lbN .'
ssh head 'scp -r ~/img worker:/home/ztxjzh/ && ssh worker "cd ~/img && docker build -t tensorfold-dsv41:v075-lbN ."'

# 7. 配置（local.sh 覆盖 config.sh 的 ${VAR:-default}）
#    IMAGE=tensorfold-dsv41:v075-lbN
#    POOL 留空（B3）、KV_SHARD=0、SPILL_GIB=0（那些 patch 未端口）
#    GRAPHS=1（0033 已端口后）

# 8. 干跑检查（关键！）
DRY_RUN=1 bash ./start.sh   # 检查两 rank --pool 对称、无未知 flag、镜像正确

# 9. 启动 + 轮询
systemctl --user reset-failed dsv41-tensorfold.service
systemctl --user start dsv41-tensorfold.service
# 轮询 /health 约 48 s，同时读 MemAvailable

# 10. 验收
#     日志必须有 "mixed pack: 27 float linears"
#     发一个请求验证输出正确
```

---

## F. 回滚

```bash
ssh head 'cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks &&
  systemctl --user stop dsv41-tensorfold.service && sleep 8 &&
  cp scripts/local.sh.pre-3a-bak scripts/local.sh &&
  systemctl --user reset-failed dsv41-tensorfold.service; sleep 2;
  systemctl --user start dsv41-tensorfold.service'
# 验证：arena 3766 MiB / 4 streams / KV_SHARD=1 / avail 8G
```

**备份链**：`local.sh.pre-3a-bak`（3a 前）、`local.sh.pre-kit-v075-bak`（换 kit 前）
**Git 冻结点**：分支 `prod-fbbc9713-frozen`、tag `pre-upstream-v075-20261124`

---

## G. 内存纪律（用户不在家，爆内存需 12 小时）

1. **任何构建前先停服**，断言 available > 100 GiB
2. 构建容器 **`--memory=40g --memory-swap=40g`**
3. **绝不并行两引擎**
4. 每次启动**轮询 health 并读 MemAvailable**
5. **不下调 `TENSORFOLD_MEMORY_RESERVE_GIB`**（引擎会吃掉 reserve，曾冻死 Spark）
6. 判定用**生产 4.8 GiB 基线**，不用空载读数
7. 驻留态**禁 ncu kernel replay**

---

## I. 公平性能对比（RDMA=1 / GRAPHS=1，两项修复后）

**口径**：decode 512 token / temp 0 / 剔首中位；prefill 用 `cold_prompt.py` 10752 tok，
未缓存 100%，`未缓存token / prefill_seconds`。

| 指标 | v0.7.5 + 5patch + MTU修复 | 我方基线 | 差 |
|---|---:|---:|---:|
| decode N=1 | **60.261** | 59.746 | **+0.9%** |
| decode N=4 聚合 | **109.803**（每流 27.451） | 109.226（27.306） | **+0.5%** |
| prefill | **2087.81** | 2085.24 | **+0.1%** |

**结论：性能持平（差值在噪声内）。** v0.7.5 底座不吃亏，且默认 **4096 行 pass**（我方 2048）。

**对比条件说明**：v0.7.5 侧为 `POOL 留空(524288) / KV_SHARD=0 / SPILL_GIB=0`，
我方基线为 `POOL=1052672 / KV_SHARD=1 / SPILL_GIB=64`。**KV 分片与 spill 尚未端口（3b）**，
所以这不是「完全体对完全体」，而是「v0.7.5 最小集 vs 我方完全体」——**在此不利条件下仍持平**。

MTP 正常：`tok/轮` 2.0157、接受率 0.5646（N=1）。

### 让分条件下的旧数据（仅供对照，勿引用为结论）

RDMA=0 + GRAPHS=0 时：N=1 52.99 / N=4 102.16 / prefill 2046.5。
**修复两项后回到 60.26 / 109.80 / 2087.81**，证明此前的 −11.3% / −6.5% 全部来自那两项让分，
**不是引擎版本差异**。

---

## H. 待补（脚本需要的未知项）
- 3b（KV 分片 + spill）的端口方案：**原 patch 建立在 0008 的 pool 上，需改造到上游原生 pool**
- C 类其余内存安全 patch（`0023` prepare 地板等）
- D 类 17 个 API patch
- `~/build-v075*` 与 `~/img-v075` 的清理策略（root 属主残留）
---

## J. 3b —— KV 分片端口（保住省 40.4% KV 内存的我方独有优势）

### J1 · 拆法：KV 分片与 spill 可分离，但**不是完全独立**

先按「只做 KV 分片」评估，冲突从 **71 块骤降到 17 块**——`lane_host.zig`(19) 与 `prompt_cache.zig`(18)
两个硬骨头**全部来自 spill**，不是 KV 分片。

**但随后发现结构性依赖**：`kv_lanes.zig` 依赖 `rank_wire.zig` + `fingerprint.zig`，
而这两个文件**由 0016（spill-blocks，8699 行）新建**。即 **KV 分片复用了 spill 引入的 rank 协商协议**，
哪怕 spill 本身关着。

**解法（已验证可行）**：不端口整个 0016，只**单独提取自包含的新文件**：

| 提取项 | 行数 | 自身依赖 | 是否自包含 |
|---|---:|---|---|
| `fingerprint.zig` | 718 | std / builtin | ✅ 完全自包含 |
| `rank_wire.zig` | 2205 | fingerprint / link / parts | ✅ 后两者已在树上 |
| `rank_identity.zig` | 549 | **spill_state** | ❌ 只在 spill+分片同开时才用，**跳过** |
| `pool.zig`（来自 0008） | 52 | std | ✅ |

`link.zig` 另需从 0016 提取 **4 个辅助函数**（上游 v0.7.5 没有）：
`readableAny`(25行) / `recvWithin`(9行) / `readAllBy`(18行) / `readable`(5行)。
**注意 `recvWithin`/`readAllBy`/`readable` 是 `Link` 的方法，必须插进 struct 内部**，
放模块级会报 `no field or member function named ...`。

### J2 · 实际端口集与冲突数

| patch | 冲突块 | 备注 |
|---|---:|---|
| 0015-dsv41-kv-shard-attn | 17 | 新建 shard/shard_ops/parts.zig + dsv41_shard.cu |
| 0018-dsv41-kv-shard-lanes | 39 | 新建 kv_parts.zig / kv_lanes.zig |
| 0022-dsv41-kv-shard-ring | 15 | RDMA ring 的行传输 |
| 0042-kv-shard-pieces | 2 | |
| **合计** | **73** | 全部已解 |

### J3 · 新增坑

**B9 · 必须逐 patch 解冲突并 `git add`，不能批量落完再解**
- 症状：`error: zig/src/.../model.zig: does not exist in index`，后续 patch 整体失败、新文件不生成
- 根因：前一个 patch 留下的 `UU`（未合并）文件挡住后一个 patch 的三方合并
- 解法：**apply → 解冲突 → `git add -A` → 下一个 apply**

**B10 · `Kind` 枚举撞号**
- 上游 v0.7.5 `fill_switch = 10`，我方 `rank_wire = 10`
- 解法：把 `rank_wire` 改 **11**，同时改 `rank_wire.zig:26` 的 `pub const lanes_kind`
- 安全性：两 rank 跑同一 build，重编号无兼容问题

**B11 · 上游把 status/noteLive 移出了 lane_host**
- v0.7.5 用 `lane_host_status.zig` 的 `status.read` / `status.noteLive`
- 我方 patch 的内联 `statusFn`/`noteLive` 要**丢弃**，但 **kv 遥测必须改挂到上游模块**：
  - `read()` 的 `out.* = .{...}` 里加 `.kv_parts = h.kv_tele,`
  - `noteLive()` 开头读 `h.kv_parts` 源，锁内写 `h.kv_tele`
- 另：`LaneHost` 的 `kv_parts` / `kv_tele` 两个字段要手工补回（易被「取上游」误删）

**B12 · 上游 fill 机制已从单流演进为多流**
- 上游 `fills: [max_streams]?Fill` + `cur`；我方 `fill: ?Fill` + `paused`
- **`filling` 字段两边重名但类型不同**（上游 `ArrayList(*Job)`，我方 `?*Job`）——必须取上游
- 原则：**取上游的多流机制，只把 `part` 字段与 KV 调用嫁接进去**

**B13 · 合并时的结构损伤（本轮实际踩到 6 次）**
取「ours + theirs 片段」时容易破坏结构，编译期才暴露：
- 字段掉到 struct 外（`engine_api.Status.kv_parts`）
- 函数缺闭合 `}`（`metrics.perPosition`）
- 局部常量重名（`native.zig` 两个 `window` → 后者改名 `served_window`）
- 重复 switch 分支（`error.BadWindow`）
- 重复常量（rdma 的 `wc_recv_rdma_with_imm` 等）
- 嵌套 `if (a) if (b) {}` 需补花括号消歧义

**B14 · 上游删掉了我方 patch 依赖的符号**
- `comm.zig` 无 `enter`/`leave` 入队门控 → `allGatherInPlace2` 按上游风格重写（`c: Comm` 值传）
- `lanes.zig` 的 `usable()` 上游叫 `refusal()`（返回 `?Refusal` 而非 bool）
  → 调用点改 `refusal(...) == null`，测试里的 `!usable(...)` 改 `refusal(...) != null`
- 取 theirs 的 `roundWins()` 后，上游内联循环里的 `const words = maskWords(...)` 与
  `var masked = ...` 需**手工补回同一函数作用域**

### J4 · 验证状态

- Mac `zig build test-cuda-host`：**427/434 通过**，剩余失败全为 macOS 固有
  （`link.zig:23` 的 Linux 专有 `c.SO.BUSY_POLL` + 3 个 benchmark 信号/计时测试）
- 补丁固化：`zig/patches-v075/0003-kv-shard-on-v075.patch`（14265 行，13 个新文件）
- Spark 构建目录：`~/build-v075-3b`（**每轮换新目录**，见 B6）

### J5 · 三个决定性排错（脚本必须内建这些断言）

**B15 · 改枚举号要改三处，漏一处即 `BadFrame`/`PairBroken`**
- `Kind` 枚举、`rank_wire.zig` 的 `lanes_kind` 常量、**`lanes.zig` 字节→Kind 解码 switch**
- 漏第三处 → rank1 `BadFrame` → rank0 `PairBroken` → 进程退出
- **`SafeAllocator leaked` 是错误后的清理噪声，不是根因**，排错时勿被带偏

**B16 · 上游拆分了 `chunk_rows` 语义**
- v0.7.5：`model.chunk_rows=4096`（pass 大小）+ `model.resume_rows=2048`（保留态网格）
- 我方 `parts.zig` / `shard.zig` 的块粒度对应 **`resume_rows`**
- 断言须写 `parts.chunk_rows == model.resume_rows`

**B17 · arena 由编译期常量定容，`TF_DS_PASS_ROWS` 改不动它** ⚠ 内存红线
- `TF_DS_PASS_ROWS` 只限制实际 pass 长度（`lanes.zig:849`）
- arena 在 `model.zig:381` 的 `prompt.Chunk.init(..., chunk_rows, ...)` 按**编译期** `model.chunk_rows` 分配
- **`chunk_rows=4096` → arena 6144 MiB → head avail 跌到 5G**（红线 4.8G）
- **必须改 `model.chunk_rows = 2048`**：arena 3872 MiB、avail 8G，性能不降反升
- 脚本断言：启动日志 `arena .* MiB used` 必须 ≤ 3900，否则拒绝切生产

### J6 · 3b 实测（RDMA=1/GRAPHS=1/KV_SHARD=1/PARALLEL=4/ROUND_ROWS=16）

| 指标 | 生产基线 | r2(无分片) | **3b(有分片)** |
|---|---:|---:|---:|
| N=1 | 59.746 | 60.261 | **59.289** |
| N=4 聚合 | 109.226 | 109.803 | **107.762** |
| prefill | 2085.24 | 2087.81 | **2061.84** |
| KV/rank | 532 MiB | 893 MiB | **532 MiB** |
| arena | 3766 MiB | 5.91 GiB | **3872 MiB** |
| avail | 8G | ~6G | **7–8G** |

**3b 运行配置**（`scripts/local.sh` 增量）：
```bash
SPILL_GIB=0                      # spill 未端口，必须 0
IMAGE=tensorfold-dsv41:v075-3b
GRAPHS=1                         # 0033 已在 r2 端口
# KV_SHARD=1 / RDMA=1 / CONTEXT=524288 / PARALLEL=4 / ROUND_ROWS=16 沿用
# POOL 由 port.sh 从 KV_SHARD=1 推导为 1052672，3b 起 TF_DS_POOL 重新生效
```

**观测缺口**：`/health` 的 `kv_shard` 仍 null（遥测链路未打通，非功能缺陷）；
启动行 "passes of up to 4096 rows" 来自 `F.prompt_rows` 家族声明，与实际 arena 定容常量不同。
