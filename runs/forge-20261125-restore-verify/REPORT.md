# 生产恢复核验（2026-11-25，E-SELF-11 离线实验后）

用户指令：「先把生产环境保证在最佳状态」。本记录是**离开前的生产状态凭据**，
全部读数按 `docs/CURRENT-BASELINE.md` 的同工具同形状口径取得，**未做任何优化改动**。

## 0. 判决

**RESULT: OK —— 生产在权威基线上，开机自启链路完整，无实验残留。**

## 1. check-boot.sh（head，Spark 项目目录下执行）

11/11 全通过，exit 0：

```
== 配置口径 ==            ok  local.sh == local.sh.authoritative
== 开机自启链路 ==        ok  Linger=yes / unit enabled / WantedBy=default.target / docker enabled
== 运行态 ==              ok  /health 200
                          ok  head image tensorfold-dsv41:base-alias present
                          ok  TF_DS_PROMPT_ALIAS=1 (patch 0068 active)
                          ok  TF_DS_L2_PREFETCH=1 (worth +6.3% single stream)
                          ok  no rejected/experiment switches in container
                          ok  avail 10 GiB (>= 6 GiB headroom)
RESULT: OK -- serving the baseline described in docs/CURRENT-BASELINE.md
```

## 2. 性能三项（同工具同形状，禁用 bench.py 跨尺寸中位）

| 指标 | 工具与形状 | 本次 | 权威基线 | 漂移 |
|---|---|---:|---:|---:|
| prefill 中位 | `cold_prompt.py <out> 28038` ×3，`/metrics` 增量 | **2102.57 tok/s** | 2084.42 | **+0.87%** |
| N=1 decode | `decode_probe.py restore-n1 256 0.0 3`，`median_excl_first` | **60.065 tok/s** | 59.265 | **+1.35%** |
| N=4 聚合 | `decode_conc.py 4 256 0.0 3`，`agg_tok_s_median` | **107.073 tok/s** | 107.621 | **−0.51%** |

**三项均在 ±1.4% 漂移带内 ⇒ 在基线。**

- prefill 三次原始：1932.21 / 2102.57 / 2105.79 tok/s。**首跑偏低是冷启动**（服务刚起 34.9 s 加载后的第一个长请求），
  按既有口径取中位。`prompt_tokens_cached_delta = 0` 三次全部为 0 ⇒ **冷前缀确实成立**；
  uncached token 10760–10764，与 10752 形状一致。
- MTP 健康：单流 `tok_per_round` 2.048、接受率 0.5848；4 路 1.8963 / 0.6798。
- 探针内存闸全程通过，请求后 `mem_after` 9.6–9.8 GiB。

## 3. 容量与身份（逐字核对，非推断）

**arena 口径澄清（本轮一度误判为回退，已澄清）**：引擎日志有两行不同的数——

```
info: the prompt pass's scratch: 25 buffers of 2458 MiB in shared regions of 900 MiB (TF_DS_PROMPT_ALIAS)
info: dsv41: pool 1052672 positions, fetch 526336 a part; ... arena 2208 MiB used
```

**2458 MiB 是 alias 前的名义 scratch，2208 MiB 才是实际 arena。**
`runs/forge-20261124-alias-audit/REPORT.md:178-179` 的 ON 臂正是这两行并列 ⇒ **配置一致，无回退**。
arena 不是 `/metrics` gauge，**只能从引擎自报行读**（本轮在 metrics 里按字节区间搜过，无匹配）。

| 项 | head | worker(spark-bbf1) |
|---|---|---|
| 镜像 | `tensorfold-dsv41:base-alias` | `tensorfold-dsv41:base-alias` |
| ImageID | `sha256:73fb55744677bd…` | 同镜像 tag |
| 镜像 Created | 2026-11-24T16:49:07+08:00 | — |
| **arena 自报** | **2208 MiB used** | **2208 MiB used** |
| Restarting / OOMKilled | false / false | false / false |
| avail | 10 GiB（探针后 9 GiB，仍高于 6 GiB 地板） | 11 GiB |

- 引擎自报：`4 streams at once`（PARALLEL=4 ✓）、`context 524288` ✓、`prompts in 2048-row chunks`（ROUND_ROWS=16 ✓）、
  `drafts: on` ✓、`compressed planes 532 MiB a rank (893 MiB replicated)`、`loaded in 34.9s`。
- `NRestarts=0`，`ExecMainStartTimestamp = Wed 2026-11-25 09:41:58 HKT` ⇒ **无崩溃重启循环**。
- fabric：`rocep1s0f1` 与 `roceP2p1s0f1` **均 PORT_ACTIVE**，另两口 PORT_DOWN（预期）。
- NFS 挂载在位（1 条），worker `192.168.100.11` ping 可达。
- 磁盘 `/dev/nvme0n1p2` 1.9T，已用 711G（**40%**），余 1.1T。

## 4. 实验残留清理

- **离线探针容器 `exl3-probe` 已删除**；head 与 worker 各只剩 1 个生产容器 `dsv41-1017-exl3-tf`。
- 容器 env 内**无任何被否决/实验开关**（check-boot 的 stray 检查 = 0）。
- 本轮所有内核改动只存在于 `runs/forge-20261125-self-decouple/src/` 与 Spark 的 `~/ptxwork/`，
  **未进入生产镜像、未改 `scripts/local.sh`、未动 `zig/patches/`**。
- `~/ptxwork/` 下的探针 `.so` 与 CSV 是报告引用的证据，**保留**（磁盘无压力）。

## 5. 仓库可复现性（地测检查）

```
tools/verify-baseline.sh
RESULT: 53/53 patches applied cleanly on pin fbbc97132c
Baseline is reproducible.
```

commit `b65e95d`（E-SELF-11）已推送 origin/main。

## 6. 离开期间的注意事项

- 服务 `Restart=on-failure` / `RestartSec=60s` / `StartLimitBurst=10` / `Linger=yes` / `enabled`
  ⇒ **重启机器与进程崩溃都会自动拉起**，就绪约 42–50 s（本次 t=50 s 时 health 200 且 avail 落到 10 GiB）。
- **`~/build-alias/` 当前是 E-SELF-04 的改动版**（备份 `.pre-gn-bak` 在旁）。
  它**不影响正在运行的 base-alias 镜像**，但**不要顺手跑 `bash ~/build-alias.sh`**，那会构建出改动版镜像。
- 若要重启生产：走 user systemd（`systemctl --user restart dsv41-tensorfold.service`），
  **绝不并行两引擎**；重启后必须在 Spark 项目目录下跑 `tools/check-boot.sh`，
  且等 `/health` 200 **且** avail 落到 10 GiB 再判定。

## 7. 回来后的第一件事（已议定，未执行）

以 `--cuda-graph-trace=node` 重采 nsys（约 10–15 min），判定 trace 里那 4 个大空隙
**216.106 / 161.824 / 152.981 / 132.638 = 663.549 ms（占窗口 13.34%，全部后接 `_hc_post`）**
是真实空隙还是 `--cuda-graph-trace=graph` 的采集伪影。

- 若真实且可归零：上界 **×1.154 → 2406 tok/s（+15.4%）**，是当前单笔上界最大、投入最小的一件事。
- 若是伪影：这 13.49% 从账上划掉，按 E-SELF-11 §7 与本对话的优先级表改投
  AllGather fp16 载荷（+6.28%，需精度闸）→ split2+消费端直读（+1.73%，位一致）→ down3 落地（+0.86%，形状兼容）。

**注意**：275 ms 固定截距与这 663 ms **可能是同一块时间的不同口径，定性前不可相加**。

## 8. 原始读数

`raw/pf-{1,2,3}.json`（prefill 三次，含 cached=0 证据）、`raw/decode-n1.json`、`raw/decode-n4.json`。