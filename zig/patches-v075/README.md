# zig/patches-v075 — v0.7.5 承重最小集（步骤3a）

**基线**：TensorFold `e100ee8b1d806a9f516d7a01cbcbf06ae98d77e1`（kit `fdde3285` 的 `ENGINE_COMMIT`，TP2 v0.7.5）
**与 `zig/patches/` 的关系**：那 52 个是对旧 pin `fbbc9713` 的；这里是对 v0.7.5 的**已解冲突**版本，**不要混用**。

## 为什么需要这一层

裸 v0.7.5 **加载不了我方 EAQ v1.1 checkpoint**：`mixed pack` / `plain_entries` / `plain_rows` 在上游 0 命中。
我方 checkpoint 有 27 个 BF16 indexer/compressor linear 必须走 cuBLAS，上游 v0.7.x 只服务纯 EXL3 2.5bpw。

##  contents

`0001-load-bearing-4-on-v075.patch` —— 一个合并补丁，含 4 个承重 patch 的**已解冲突**结果：

| 原 patch | 作用 | 落地情况 |
|---|---|---|
| `0011-cuda-budget-parked-pages` | 启动预算计入 per-CPU parked pages | 2 文件冲突，已解 |
| `0024-mixed-pack` | EAQ v1.1 的 27 个 BF16 linear 走 cuBLAS | **三方合并全干净** |
| `0025-widen-bf16` | BF16 源精确展宽 f32 | **全干净** |
| `0026-plain-rows` | plain linear 定形，保 token bits 稳定 | 2 文件冲突，已解 |

## 已丢弃

**`0008-pool-window`**：v0.7.5 原生支持 `--pool` / `--context` / `TF_DS_WINDOW`，语义等价
（我方 `POOL` ↔ 上游 `--pool`；我方 `WINDOW` ↔ 上游 `--context`；`ctx_cap = min(context, pool)`）。
连带丢掉 `pool.zig` 与其在 model/native 的冲突。

**`0012-dsv41-base-fixes`**：其一半是 0010 的 Linux-only prctl 配套（0010 未纳入），另一半是 Engram 读-ahead 修复，
3a 不需要；留待 3b 评估。

## 冲突解法要点（7 块 / 4 文件）

- `cuda_memory.zig`：`procText`（上游）与 parked 三件套（我方）是不同函数，**并存**
- `cuda.zig` ×3：① `procText` 已移到 budget 模块，改限定名；② **保留上游更新的准入逻辑**
  （`learn_bytes` / `peer_room` / `free_streams`），只加回 parked 日志；③ 测试区两边并存，
  但**删除重复的 `chipClass` 测试**（上游已移至 `cuda_device.zig` 且有逐字相同的一份）
- `model.zig`：**只取 `m.eng.plain_rows = o.plain_rows`**，丢弃 theirs 侧混入的 KV_SHARD 代码（属 3b）
- `native.zig` ×2：保留上游 `streams=32` / `16..64` 等文档与错误串，追加 `PLAIN_ROWS` / `BadPlainRows`

## 验证状态

- **Mac（zig 0.17.0）主机测试 426/433 通过**，我方改动全部编译通过
- 剩余失败均归因 macOS 平台：`link.zig` 的 `SO_BUSY_POLL` 为 Linux 专有（**该文件未被本次改动触及**，
  `git diff` 为空）；3 个 benchmark 信号/计时测试
- **尚未在 Spark 上构建，尚未加载 checkpoint 验证** —— 这是 3a 的真正验收，未做

## 用法

```bash
git worktree add /tmp/tf-v075 e100ee8b1d806a9f516d7a01cbcbf06ae98d77e1
cd /tmp/tf-v075 && git apply zig/patches-v075/0001-load-bearing-4-on-v075.patch
```
