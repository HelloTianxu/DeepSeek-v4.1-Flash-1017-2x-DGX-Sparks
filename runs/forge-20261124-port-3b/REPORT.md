# E-PORT-3b · 把 KV 分片端口到 v0.7.5

**结论：目标达成。** 我方独有的 KV 分片（省 40.4% KV 内存）已在 v0.7.5 上跑通并实测验证，
内存回到生产基线水平，性能代价约 1%。

- 工作树最终 commit：`296c429f2`（基 `32596a37b` = r2）
- 补丁：`zig/patches-v075/0003-kv-shard-on-v075.patch`（14280 行，13 个新文件）
- 镜像：`tensorfold-dsv41:v075-3b`（两节点）

---

## 1. 实测对比（RDMA=1 / GRAPHS=1 / KV_SHARD=1 / PARALLEL=4 / ROUND_ROWS=16）

| 指标 | 生产基线（旧引擎+分片） | r2（v0.7.5 无分片） | **3b（v0.7.5+分片）** | 3b vs 基线 |
|---|---:|---:|---:|---:|
| N=1 decode tok/s | 59.746 | 60.261 | **59.289** | −0.8% |
| N=4 聚合 tok/s | 109.226 | 109.803 | **107.762** | −1.3% |
| N=4 每流 tok/s | 27.307 | 27.451 | **26.941** | −1.3% |
| prefill tok/s（中位） | 2085.24 | 2087.81 | **2061.84** | −1.1% |
| **KV 压缩面 / rank** | 532 MiB | 893 MiB（复制） | **532 MiB** | **省 40.4% ✓** |
| arena | 3766 MiB | 5.91 GiB | **3872 MiB** | +106 MiB |
| head avail | 8 G | ~6 G | **7–8 G** | 持平 |

口径：prefill = `未缓存输入token / server.prefill_seconds`，10752 token，三次 2061.84 / 2006.38 / 2064.11。
N=1 三次 57.914 / 59.285 / 59.294（取除首次外中位）。N=4 三次聚合取中位。
MTP 正常：tok/轮 2.0157、接受率 0.5646（N=1）。

**日志铁证**（两 rank 一致）：
```
info: mixed pack: 27 float linears
info: rank cache: 95.1 GiB of expert entries read whole in 21.2 s
info: dsv41: pool 1052672 positions, fetch 526336 a part; a conversation may use up to 524288 tokens;
      compressed planes 532 MiB a rank (893 MiB replicated); arena 3872 MiB used
```

---

## 2. 端口集与冲突

只做 KV 分片（不做 spill），冲突从 71 块降到 17 块——`lane_host.zig`(19) 与 `prompt_cache.zig`(18)
两个硬骨头**全部来自 spill**。

| patch | 冲突块 |
|---|---:|
| 0015-dsv41-kv-shard-attn | 17 |
| 0018-dsv41-kv-shard-lanes | 39 |
| 0022-dsv41-kv-shard-ring | 15 |
| 0042-kv-shard-pieces | 2 |
| **合计** | **73** |

### 2.1 结构性依赖：KV 分片复用 spill 的协商协议

`kv_lanes.zig` 依赖 `rank_wire.zig` + `fingerprint.zig`，两者**由 0016（spill-blocks）新建**。
解法不是端口整个 0016，而是**单独提取自包含新文件**：

| 提取项 | 行数 | 依赖 | 判定 |
|---|---:|---|---|
| `fingerprint.zig` | 718 | std/builtin | 自包含 ✓ |
| `rank_wire.zig` | 2205 | fingerprint/link/parts | 后两者已在树 ✓ |
| `rank_identity.zig` | 549 | **spill_state** | 只在 spill+分片同开时用，**跳过** |
| `pool.zig`（0008） | 52 | std | 自包含 ✓ |

`link.zig` 另需 4 个辅助函数：`readableAny`(25) / `recvWithin`(9) / `readAllBy`(18) / `readable`(5)。
后三个是 `Link` 的**方法**，必须插进 struct 内部。

---

## 3. 三个决定性排错

### 3.1 `PairBroken` / `BadFrame` ← 枚举改号漏了第三处

上游 `Kind.fill_switch = 10` 与我方 `rank_wire = 10` 撞号，改 `rank_wire = 11`。
但改号涉及**三处**，我只改了前两处：

1. `lanes.zig` 的 `Kind` 枚举 ✓
2. `rank_wire.zig:26` 的 `pub const lanes_kind` ✓
3. **`lanes.zig:1550` 的字节→Kind 解码 switch ✗ ← 漏了**

后果：rank0 发字节 11，rank1 解码 switch 无此分支 → `else => return error.BadFrame`
→ rank1 停止跟随 → rank0 报 `PairBroken` → 进程退出（伴随 SafeAllocator leak 噪声，**leak 是果不是因**）。

补 `11 => .rank_wire,` 后协商通过。

### 3.2 comptime 断言失败 ← 上游拆分了 chunk_rows 语义

上游 v0.7.5 把我方单一的 `chunk_rows=2048` 拆成两个：
- `model.chunk_rows = 4096` —— prompt pass 大小
- `model.resume_rows = 2048` —— **保留态网格**（kept state 落点）

我方 `parts.zig` 的 `chunk_rows` 用于 `at % chunk_rows == 0`（保留态判定）与放置对齐，
**对应的是 `resume_rows`**，值 2048 本身正确。断言应比 `model.resume_rows`。
（`growth_rows` 两边本来就相等，是佐证。）

### 3.3 arena 6144 MiB / avail 跌到 5G ← 必须改编译期常量

**`TF_DS_PASS_ROWS=2048` 无效**：该 env 只限制实际 pass 长度（`lanes.zig:849`），
**arena 由编译期 `model.chunk_rows` 定容**（`model.zig:381` 的 `prompt.Chunk.init`）。

改 `model.chunk_rows = 4096 → 2048` 后：arena **6144 → 3872 MiB**，avail **5 → 8 G**。
性能未受损（N=4 反而从 106.691 升到 107.762）。

> 这直接回答了此前「chunk4096 开不起」的方向：**4096 行 pass 在我方内存预算下不可行**，
> 上游能开是因为它不做 KV 分片时池布局不同。

---

## 4. 未决 / 待补

- **`/health` 的 `kv_shard` 字段仍为 null**：分片本身已由启动日志与内存数证实生效，
  但遥测链路（`lane_host_status.noteLive` → `h.kv_tele` → `read`）未打通，属**观测缺口非功能缺陷**。
- **启动行仍报 "passes of up to 4096 rows"**：来自 `F.prompt_rows` 家族声明，与 `model.chunk_rows` 不是同一常量，
  仅上报口径滞后，实际 arena 已按 2048 分配。
- spill 四件套（0013/0014/0016/0021）未端口；`SPILL_GIB` 必须为 0。
- 生产多路（head 4.8 GiB）下未复验，本轮全部为空载 avail 7–8 G。
- 是否切生产、验收标准，待用户决定。

---

## 5. 回滚

```bash
cp scripts/local.sh.pre-3b-bak scripts/local.sh
systemctl --user stop dsv41-tensorfold.service && sleep 8
systemctl --user reset-failed dsv41-tensorfold.service
systemctl --user start dsv41-tensorfold.service
# 验证：arena 3766 MiB / KV_SHARD=1 / avail 8G
```