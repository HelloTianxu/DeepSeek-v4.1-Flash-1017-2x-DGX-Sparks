# E-FEED-02：pinned upload 改造 + 同场 A/B —— 改造否决，但**因果假说被判决性否证**

> # ⛔ 重大更正（2026-11-25 当日）：**§4 的「修正后解释」作废；否决的真因是空隙早已不存在**
>
> 本轮沿用了 E-FEED-01 的 11-23 trace，而那份 trace 处于 **`ENGRAM_AIO=1`** 时代。
> **11-24 的 E-ENGRAM-01 已落地 `AIO=0`**，`01-发现与结论.md` §6.5 的零注入 PM 采样直接测得
> 当前配置 **>10 ms 空闲段 0 个、最长 0.2 ms、空闲 0.04%、GPU 繁忙度 99.96%**。
>
> ⇒ **pinned 改造无收益的真正原因是：它要消除的那 663.5 ms 空隙，在 11-24 就已经被消除了。**
> 不是 §4 写的「节流不是瓶颈的原因」，更不是「磁盘读绝对时长超过重叠窗口」——**那两条解释都作废**。
>
> §6.5 的交叉验证已给出正确机制：关 AIO 省的 1.2076 s 中 **1.1167 s（92.5%）来自空隙消除**，
> 真实 GPU 工作只减 **0.8%** ⇒ **engram 的读没有消失，是从阻塞 GPU 变成与 GPU 工作重叠**。
> 这正解释了为什么本轮离线测得 gather 671–696 ms 却不构成卡点。
>
> **另作废**：§4 末「三条 IO 侧干预全负都能用 page cache 统一解释」是过度归纳。
> AIO 的真因**既有记录早已查明**（§6.4）：**单线程 O_DIRECT 每批 2048 读经 blk-mq 落同一硬件队列（1023 tags），
> `io_submit` 阻塞等 tag 回收**；抖动 15.6%→0.9% 是队列饱和的指纹。与 pinned 的锁页挤占 cache 是**两个不同机制**，不可统一。
>
> **本轮仍然有效的产出**：
> ① **机制判决探针**（`tools/pin-gate.cu`）——24 MiB + GPU 带负载下 pageable host API **81.879 ms** vs pinned **0.002 ms**，
> 独立证实了「pageable 的 stream-ordered copy 是隐式节流点」，并证明历史 4096 B 空载 gate 的结论不可外推；
> ② A/B 工程范式（同一二进制翻 env、身份断言锚到本次 restart、trap 自动还原配置）；
> ③ 手册 §7.18「否证因果链必须单独验证干预是否生效」的方法本身正确——**只是本轮被否证的那条链，描述的是历史配置**。
>
> **方法论失误**：违反 §7.2（先查已有实测批注）。已立 §7.19：**引用历史 trace 前必须核对期间全部已落地配置变更**。

日期 2026-11-25 · 生产已恢复基线（`check-boot.sh` **RESULT: OK**）
前序 `runs/forge-20261125-feed-root/REPORT.md`（E-FEED-01）

---

## §0 判决表

| 待判命题 | 判决 | 判决性读数 |
|---|---|---|
| pinned 能提升 prefill | **否决** | 三轮配对 **−0.742% / −0.141% / −0.210%**，**全负同向**，均值 **−0.364%** |
| pinned 改造是否真的解除了 host 节流 | **是，完全有效** | 24 MiB + GPU 带负载：host API **81.879 ms → 0.002 ms**（约 **40939×**） |
| **「upload 节流导致 GPU 空隙」**（E-FEED-01 §3 的因果链后半段） | **判决性否证** | 机制干预有效 4 万倍，而目标指标不动且略负 ⇒ 该机制**不是**空隙的原因 |
| pinned 是否伤 decode | 无方向性 | 配对 **+1.603% / +0.088% / −0.377%**，不同向 ⇒ 噪声 |

**这是本轮最有价值的产出**：不是「pinned 没用」，而是**干净地排除了一条因果路径**。
E-FEED-01 的空隙归因（engram 磁盘读）仍成立，但「因为 upload 钉住主线程所以预读起不来」这个**中间环节被推翻**。

## §1 改动设计

**为什么不改 `EngramHost.init` 签名**：它有 **6 处调用点**，其中 4 处是 `engram_lead.zig` 的单元测试（无 CUDA driver）。
改签名会波及测试。改为新增结构体方法 `EngramHost.pinRows(eh, d)`，只在 `model.zig` 的生产路径调用一次，测试路径零影响。

**为什么可以直接替换 slice 而不拷旧内容**：init 后这些缓冲未初始化，`engramAhead1` / `decodeRows` 都是先写满再 upload，不存在先读后写。

**锁页量**：`eh.rows` 24 MiB + `ahead_rows[0..1]` 各 24 MiB = **72 MiB**，远低于 4.8 GiB 红线。
`EngramHost` 只在 `Model.open` 构造一次（`model.zig:205-222` 的字段清零在 `pub fn open` 内），**不存在每请求泄漏**。

**开关**：`TF_DS_ENGRAM_PINNED`，默认关。`start.sh:291` 把非黑名单的 `TF_DS_*` 自动传两个 rank，
`TF_DS_ENGRAM_PINNED` 不匹配黑名单的 `ENGRAM=`（要求紧跟 `=`）⇒ 两 rank 都收到。

**位一致性**：只换内存来源，不动任何数值。

## §2 同场 A/B（同一二进制，只翻 env）

镜像 `tensorfold-dsv41:pinned-ab`（FROM `base-alias`，继承 alias patch，**不动 cubin**），两节点均构建。
**对照臂同场重测，未复用任何历史基线数值。** A/B/A/B/A/B 交替三轮。

**身份断言**（每臂都验，只看本次 restart 之后的日志）：
`PINNED=0` → `page-locked` 日志 **0** 行；`PINNED=1` → **1** 行。六臂全部符合。

prefill（`cold_prompt.py 28038` 字符 = 10752 token 冷前缀，`cached_delta` 全 **0**，每臂 3 次）：

| 轮 | p0 中位 | p1 中位 | 配对差 | 剔冷启动后配对差 |
|---|---:|---:|---:|---:|
| r1 | 2101.02 | 2085.44 | **−0.742%** | −0.777% |
| r2 | 2093.93 | 2090.97 | **−0.141%** | −0.179% |
| r3 | 2098.42 | 2094.02 | **−0.210%** | −0.229% |

**三轮全负、同向** ⇒ 按小效应纪律（≲1% 看配对差是否同向）判为**真实的小幅负效应**，不是噪声。
每臂 run1 均为冷启动（1927–1941），run2/run3 稳定，两种口径结论一致。

decode N1（`decode_probe.py 256 0.0 3`，排除首 rep）：59.005→59.952 / 59.932→59.984 / 60.047→59.820，
配对 **+1.603% / +0.088% / −0.377%**，**不同向 ⇒ 噪声，无回退**。

## §3 机制判决探针（未停服，生产容器内）

`tools/pin-gate.cu`：**24 MiB 真实尺寸 + GPU 带负载**（spin 81.419 ms 占满 48 SM），测 host 侧 API 返回时长。

| 源 | host API 中位 | API + sync 中位 |
|---|---:|---:|
| pageable | **81.879 ms** | 81.879 ms |
| pinned | **0.002 ms** | 81.858 ms |

pageable 的 API 时长 ≈ 整个 spin ⇒ **host 被 GPU 消费速度完全限速**；pinned 立即返回 ⇒ **节流彻底解除**。
两者 API+sync 相同（都要等 GPU），印证 E-FEED-01 §2「那 2642 ms 是有效计算不是浪费」。

**对照历史 gate 的缺陷**：`pageable-copy-gate-v2` 用 **4096 B 空载**测得 2.284 vs 1.831 µs 判「可忽略」。
**小包空载完全暴露不出本效应**，其结论不可外推（已写入手册 §7.17c）。

## §4 因果否证的意义与方向修正

E-FEED-01 §3 的链条是：
`pageable upload 钉住主线程 → 无法更早 spawn 预读 → 下一 chunk 在 th.join() 等磁盘 → GPU 空隙`

本轮把**第一环彻底打断**（81.879 ms → 0.002 ms），**最后一环却不动**。⇒ **第一环不是最后一环的原因。**

**仍然成立的部分**：空隙真实（663.5 ms / 13.34%）、位置紧贴 engram 层、engram gather 冷读成本 671–696 ms。
**被推翻的部分**：把空隙归给「host 被 upload 钉住所以预读起不来」。

**修正后的解释（假说，未验证）**：卡点是**磁盘读的绝对时长超过可用重叠窗口**，与 host 何时空出手无关。
预读线程即使立刻启动，671 ms 的随机读仍可能盖不住两层之间的 GPU 工作量。

**一个统一的旁证**：三条 IO 侧干预全部负收益，且都能用「page cache 命中」解释——
- **AIO / O_DIRECT**：孤立冷读快 18.4%（548 vs 671 ms），但**绕过 page cache**，端到端反而更慢（历史 1773 vs 2118 tok/s）
- **pinned**：锁页 72 MiB **挤占** page cache 可用量，本轮 −0.364%
- 表 203 GB 而 `buff/cache` 仅 9 GB ⇒ 命中率低但非零，任何减少 cache 的改动都在负方向

⇒ **下一步应转向「提高 engram 读的 cache 命中 / 减少绝对 IO 量」，而不是继续改 host 调度时机或换 IO 路径。**
**注意这与「先把内存吃满」的目标存在张力**：吃满显存/内存会进一步挤占 page cache。此张力**尚未量化**，是下一个该测的东西。

## §5 自纠

1. **两次编译期错误，均被预筛拦下**（§7.8 的价值）：
   ① `pinRows` 插在了 `EngramHost` 结构体内（锚点 `useLead` 有 4 空格缩进），却按顶层函数调用 `prompt.pinRows(eh,d)` ⇒ 改为 `eh.pinRows(d)`；
   ② `std.c.getenv` 返回 `[*:0]u8` 不是 slice ⇒ 照既有 `TF_DS_PROMPT_ALIAS` 范式用 `std.mem.span(v)`。
2. **A/B 脚本的身份断言原有 bug**：`journalctl --since "-3min"` 会覆盖**上一臂**的日志，导致 `PINNED=0` 臂误判有 `page-locked`。改为取本次 restart 的精确时刻。跑前修掉。
3. **`decode_probe.py` 输出是 JSONL 不是单个 JSON**，首版解析脚本 `json.load` 报 `Extra data`。原始文件已存盘，改逐行解析后补齐 decode 结论。
4. **编译前必须停服**：构建容器 `--memory=40g`，而运行态 avail 仅 8 GiB。先停服（avail 回到 **118 GiB**）再编译，避免撞内存红线。
5. **不能以 Mac 的 `_upstream/tf` 为改动基准**：其 HEAD 是 `ea64a870d`，与 Spark `build-alias` 的 `prompt.zig` sha256 不同（`0a57d971` vs `f3511c4d`），且 Mac 版**没有** `TF_DS_ENGRAM_LEAD` 机制。改动一律以 Spark 侧那棵编出生产镜像的树为准。
6. **`build-alias` 是 E-SELF-04 改动版**：复制成独立树 `build-pinned` 后用 `.pre-gn-bak` 还原 `exl3_experts.zig` / `exl3_experts2d.zig`，原树未动。

## §6 恢复核验

- `scripts/local.sh` 由脚本 `trap EXIT` 自动还原（`IMAGE=tensorfold-dsv41:base-alias`，无 `ENGRAM_PINNED`）
- 重启后就绪 **t=33s**，`check-boot.sh` **RESULT: OK**（含 `TF_DS_PROMPT_ALIAS=1`、`TF_DS_L2_PREFETCH=1`、无实验开关、avail 9 GiB）
- 原树 `~/build-alias` 与基线产物 `~/build-alias-out` **未改动**；新增 `~/build-pinned{,-out}`、`~/img-pinned`、镜像 `pinned-ab`（保留供复现，未进生产配置）

## §7 复现

```bash
# 改动（幂等，assert 每处 count==1）
python3 tools/patch-engram-pinned.py ~/build-pinned
bash ~/build-pinned.sh            # 需先停服；BUILD_OK + EXIT=0
# A/B（自动备份/还原 local.sh，含身份断言）
bash tools/ab-engram-pinned.sh    # ROUNDS=3
# 机制判决（未停服，生产容器内）
docker cp tools/pin-gate.cu dsv41-1017-exl3-tf:/tmp/
docker exec dsv41-1017-exl3-tf bash -c \
  "/usr/local/cuda/bin/nvcc -O3 -arch=sm_121 -o /tmp/pin-gate /tmp/pin-gate.cu && /tmp/pin-gate"
```

## §8 原始读数

- `raw/feed-ab/`（24 件）：`pf-r{1,2,3}-p{0,1}-{1,2,3}.json` 18 件 + `dec-r{1,2,3}-p{0,1}.json` 6 件（JSONL）
- `raw/feed-ab.log`：A/B 全程 stdout（含每臂身份断言与就绪时间）
- `raw/paired-summary.json`：配对差汇总与判决字段
- `raw/pin-gate-24mib.txt`：机制判决探针读数与解读