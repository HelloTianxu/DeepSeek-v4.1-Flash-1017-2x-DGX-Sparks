# E-FEED-01：prefill「喂不饱 GPU」的根因归因

日期 2026-11-25 · 生产**未停服、未改配置、未重采**，全部结论来自既有 trace 的零成本重算 + 一次纯 host 离线基准
基线口径 `docs/CURRENT-BASELINE.md`（prefill 中位 **2084.42 tok/s**）

---

## §0 判决表

| 待判命题 | 判决 | 判决性读数 |
|---|---|---|
| 663.5 ms 空隙是 `--cuda-graph-trace=graph` 的采集伪影 | **否证** | 全程 `cuGraphLaunch` 仅 **3** 次；4 个空隙内严格内含 API 数 = **1/1/1/0**（各 37 µs 量级） |
| 6 条巨型 HtoD（2642.7 ms = 窗口 53.1%）是可回收的浪费 | **否证（自纠）** | 期间 GPU 忙 **100.0/100.0/100.0/100.1/99.8/100.1 %** ⇒ host 被隐式节流，这段时间是有效计算 |
| 空隙是真实 GPU 空闲 | **成立** | 空隙按定义 GPU 忙 0%，且内部无 CUDA API 提交 |
| 卡点在 GPU 内核 | **否证** | 卡点在 **host 侧 engram 表读**；49152 行 buffered gather 实测 **671.19 / 695.89 ms** |
| 换 IO 路径（O_DIRECT / AIO）能救 | **不成立** | O_DIRECT 更慢；AIO 孤立快 18.4% 但 `o.engram_aio` 一个开关同管 decode+prompt，历史端到端否决完整 |

**一句话**：GPU 不是算不动，是**主线程在 engram 层的 `th.join()` 上等磁盘**，等的那段就是空隙。可回收上界 **13.34%**，不是 53%。

---

## §1 否证一：采集伪影假说

原计划以 `--cuda-graph-trace=node` 重采（10–15 min + 停服）。先做零成本验证，结果直接否证了立项前提：

- 全 api 里 graph 相关只有 `cuGraphLaunch` **3**、`cuGraphInstantiateWithFlags` 1、`cuGraphDestroy` 1、`cuGraphExecDestroy` 1。3 次 launch 不可能造成 4 个 130–216 ms 的空隙。
- 四个空隙**严格内含**的 API 数为 1/1/1/0，唯一那条是 `cuLaunchKernelEx`（36.9–37.9 µs）。即 216 ms 里只有 37 µs 有 API 活动。
- 全部 18264 个 kernel 都在**单一 stream 13**，不存在「侧流 kernel 漏记」。

⇒ 空隙不是漏记，重采 node 粒度**没有必要**。原计划撤销。

## §2 否证二：巨型拷贝不是浪费（本段最重要的自纠）

req2 内 `cuMemcpyHtoDAsync_v2` 共 **505** 条，host 侧累计 **2661.7 ms**：

| 分档 | 条数 | 合计 |
|---|---:|---:|
| <100 ms | 499 | **18.993 ms** |
| ≥100 ms | **6** | **2642.664 ms** |

字节数由源码断言反解，不猜：`prompt.zig:1357` 断言 `m * t.row_w == n * ew.k`，`upload` 传 `m * t.row_w * 2`
⇒ 字节 = `chunk_rows × ew.k × 2` = 2048 × 6144 × 2 = **25,165,824 B = 24.0 MiB**。
（`ew.k=6144` 来自 `engram.wkv.weight` shape `[25600, 6144]`；`row_w=256` 来自 `embed.weight` shape `[384016682, 256]`；反解 k = 6144/256 = **24**。）

表观带宽只有 **39.2–126.3 MB/s**，看着像拷贝病了。但判决要看同期 GPU：

| upload @ms | 时长 ms | GPU 忙 | 表观 MB/s |
|---:|---:|---:|---:|
| 493.3 | 375.3 | **100.0%** | 67.1 |
| 1366.6 | 605.0 | **100.0%** | 41.6 |
| 2463.5 | 599.4 | **100.0%** | 42.0 |
| 3533.1 | 642.3 | **100.1%** | 39.2 |
| 4289.3 | 199.3 | **99.8%** | 126.3 |
| 4492.3 | 221.4 | **100.1%** | 113.6 |

⇒ **pageable 源的 stream-ordered async copy 是隐式节流点**：driver 分块 staging，host 被 GPU 消费速度反向限速。这 2642.7 ms 里 GPU 一直满载，**它是有效计算时间，不是浪费**。表观带宽在这种情况下是假象，不可当带宽读数用。

**若不做这一步，就会立一个 53% 的虚假上界。** 真实可回收只有空隙的 13.34%。

## §3 空隙结构与源码因果链

四个空隙（>1 ms 的空隙恰好只有这 4 个）：

| 空隙 | ms | 区间 ms | 前一个 kernel | 后一个 kernel |
|---|---:|---|---|---|
| 1 | 216.106 | [43.366, 259.473] | `AllGather_RING_LL` grid 64 | `_hc_post` 1.146 ms |
| 2 | 161.824 | [1199.558, 1361.381] | 同上 | `_hc_post` 1.142 ms |
| 3 | 152.981 | [2304.975, 2457.956] | 同上 | `_hc_post` 1.147 ms |
| 4 | 132.638 | [3395.028, 3527.666] | 同上 | `_hc_post` 1.146 ms |
| **合计** | **663.549** | **13.34% 窗口** | | |

四次模式完全一致，且 `_hc_post` 时长几乎相同（1.142–1.147 ms）⇒ 是同一个结构性位置，不是随机抖动。

源码链（`_upstream/tf`，pin 对应工作树）：

1. `prompt.zig:1332` —— `engramApply` **开头**就调 `tri_basic.hcPost` ⇒ 这就是 `_hc_post`。**空隙紧接在 engramApply 之前**。
2. `model.zig:709-712` —— 每个 engram 层，主线程 `th.join()` **硬等预读线程**：
   ```zig
   if (ahead) |th| { th.join(); ahead = null; }
   ```
3. `prompt.zig:1350-1355` —— 预读未命中时现场做：`hasher.hashes()` → `pool.gather()`（磁盘 pread）→ `engram.decodeRows(..., 8)`（8 线程解码）。
4. `prompt.zig:1359` —— `upload(e, ch.eb, rows.ptr, ...)`，注释明写 **"Pageable host memory is staged before the call returns."**
5. `model.zig:720-727` —— **下一 chunk 的预读只在最后一个 engram 层之后才 spawn**，且受 `m.next_rows` 限制。

⇒ 闭环：**pageable upload 把主线程钉住 375–642 ms → 主线程无法更早推进 chunk 循环、更早 spawn 下一 chunk 预读 → 下一 chunk 到 engram 层时 `th.join()` 现场等磁盘 → GPU 队列排空后空闲 130–216 ms。**

## §4 离线判决实验（零停服）

镜像自带 `tf-dsv41-engram-check`（`engram_check_main.zig`），纯 host、不占显存，可与生产并存。
按生产真实行数 **49152 = 2048 × 24** 跑：

```
tf-dsv41-engram-check /engram --rows 49152 --seed 7
```

| 层 | buffered（生产路径） | O_DIRECT | AIO | 字节一致 |
|---|---:|---:|---:|---|
| 1 | **671.19 ms** | 716.14 | 548.05 | ✓ |
| 14 | **695.89 ms** | 709.37 | 553.74 | ✓ |

`all_equal: true`；`real 3.932s / user 0.601s / sys 6.880s` ⇒ **内核态 IO 时间主导，确是 IO 受限**。
跑前 `buff/cache` 仅 9 GB（表 203 GB）⇒ 冷读，不是缓存命中的乐观读数。

表规模：`embed.weight [384016682, 256]` FP8 E4M3 + `embed.scale [384016682, 8]` E8M0，两个 safetensors 各 ~101.5 GB。

**成本 671–696 ms 远大于空隙 130–216 ms**，正说明预读**已经隐藏了大部分**，空隙是**未被重叠的残余**。（本工具 pool 32 线程，生产 64 线程，生产更快，方向不变。）

**证据强度说明（不可当直证）**：以上是「位置吻合 + 机制存在 + 量级足够」的三重旁证，**尚未直接测到 `th.join()` 的实际等待时长**。直证需 `RANK1_PROFILE=1` 的 `prefill_engram_ms.rows` 段（`lanes_main.zig:161/281` 打印），或 pinned 改造的 A/B 反证。

## §5 上界与下一步

Amdahl（底 2084.42 tok/s，窗口 4972.654 ms）：空隙 663.549 归零 ⇒ ×**1.1540** ⇒ **2405.4 tok/s（+15.4%）**。
**这是上界，不是预期收益**；且**禁止**与 AllGather fp16(+6.28%)、split2(+1.73%) 等连乘。

按投入产出排序：

1. **pinned host buffer 替代 pageable upload**（`eh.rows` 由 `gpa.alloc` 改 `cuMemHostAlloc`）。机制最直接：解除 host 节流 ⇒ 主线程立刻返回 ⇒ 预读更早启动。改动有界，**位一致可验**（只换内存来源，不动数值）。
2. **提前 spawn 下一 chunk 预读**（放宽 `model.zig:720` 的「仅最后一个 engram 层后」条件），让磁盘读跨 chunk 流水。
3. **不重试**：O_DIRECT（更慢）、AIO（端到端已否决，且一个开关同管两侧）。

**注意既有 gate 的设计缺陷**：`pageable-copy-gate-v2.txt` 用 **4096 B 小包**测得 pageable 2.284 µs vs pinned 1.831 µs，据此判「差异可忽略」。**小包完全暴露不出大包的分块节流效应**——24 MiB 下 host 被钉住 600 ms。该 gate 的结论只在 KB 量级成立，不可外推。

## §6 自纠

1. **区间重叠 ≠ 严格包含**。首版用 `x[0]<g1 and x[1]>g0` 筛空隙内 API，把空隙**结束后**的 `cuLaunchKernelEx`/`cuMemcpyHtoDAsync` 误算进内部，据此差点编出「空隙内有拷贝」的故事。改严格包含后真相是「几乎零 API」。
2. **表观带宽不可当带宽读数**。2642.7 ms / 24 MiB 算出 40 MB/s，看着像重大缺陷；实测同期 GPU 忙 100% ⇒ 是节流假象。**先判同期 GPU 忙闲，再谈带宽。**
3. **原重采计划基于错误假说**。`--cuda-graph-trace=node` 是为「graph 伪影」准备的，该假说被 3 次 `cuGraphLaunch` 直接否证 ⇒ 计划撤销，省下一次停服。
4. **小效应 gate 不可外推到大形状**（见 §5 末）。
5. 又一次踩到已记录的坑：ssh 单引号内嵌 python heredoc 时 f-string 里用 `\"` 会 SyntaxError。改用无引号 `print(k, v)`。

## §7 复现

```bash
# 空隙与拷贝定价（零成本，读既有 trace）
python3 tools/feed-gap-attr.py runs/forge-20261123-steady-trace/timeline.json --req req2

# engram 读成本（零停服，纯 host，生产容器内）
ssh ztxjzh@192.168.1.118 'docker exec dsv41-1017-exl3-tf \
  bash -c "tf-dsv41-engram-check /engram --rows 49152 --seed 7"'
```

源码坐标：`prompt.zig:1326/1332/1341/1346/1350-1359/1371`、`model.zig:382/384-391/684/690/709-712/719-727`、`engram_check_main.zig:1-60`、`lanes_main.zig:150-161/281`。

## §8 原始读数

- `raw/gap-attr.json` —— 脚本输出（空隙、严格内含 API、巨型拷贝与同期 GPU 忙占比、Amdahl）
- `raw/engram-check-49152.txt` —— 三条读路径逐层读数 + `time` + 跑前 `free -g` 口径

关键标量：span 4972.654 ms / kernel busy 4301.959 ms / 空隙 663.549 ms(13.34%) / `cuGraphLaunch` 3 / 巨型拷贝 6 条 2642.664 ms / 小拷贝 499 条 18.993 ms / 每次 upload 24.0 MiB / kernel stream 全 13。