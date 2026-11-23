# E-DRAM-01 停服前快照与唯一恢复路径

日期 2026-11-23，Asia/Shanghai。授权原话（本轮）：**"先A后B吧，这个环境你是可以停机操作研究的 我可以暂时用云上模型"**。
上一轮授权："这个TP2环境你可以停服测试，现在没有在使用。注意，只要不破坏推理环境就可以"。

## 1. 停服前快照（实测）

| 项 | 值 |
|---|---|
| `systemctl --user is-active dsv41-tensorfold.service` | `active` |
| 容器 | `dsv41-1017-exl3-tf` Up 27 minutes |
| `curl http://127.0.0.1:8888/health` | `{"status": "ok"}` |
| 镜像 | `tensorfold-dsv41:exl3-tp2-8348338a8d90` |
| 镜像 ENTRYPOINT | `/opt/nvidia/nvidia_entrypoint.sh`（banner 后 exec "$@"） |
| unit ExecStart | `/usr/bin/bash ./start.sh` |
| unit ExecStop | `/usr/bin/bash ./stop.sh` |
| WorkingDirectory | `/home/ztxjzh/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks` |
| Restart / RestartSec | `on-failure` / `60s` |
| StartLimitBurst / Interval | `3` / `30min` |

配置未改：`scripts/local.sh` 保持 TP2、CONTEXT=524288、SPILL_GIB=64、KV_SHARD=1、
EAQ_V1_1=1、ABLIT=1、API_KEY=1、PREFILL_INTERLEAVE_MS=0。

## 2. 唯一恢复路径

```bash
ssh ztxjzh@192.168.1.118
# 若本轮临时容器还在，先按项目方式停（会一并停 worker）
cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks && ./stop.sh
# 恢复生产（唯一入口，勿直接跑 start.sh：systemd 持 start.lock）
systemctl --user start dsv41-tensorfold.service
```

若 30 分钟内失败 3 次触发 StartLimit：`systemctl --user reset-failed dsv41-tensorfold.service` 后再 start。
**不修改 unit 的 StartLimitBurst/Interval。**

## 3. 验收标准（三条全过才算恢复）

1. `systemctl --user is-active dsv41-tensorfold.service` == `active`
2. `docker ps` 显示 `dsv41-1017-exl3-tf` Up（head 与 worker 各一）
3. `curl -s http://127.0.0.1:8888/health` 返回 `{"status": "ok"}`

## 4. 本轮对生产环境的改动边界

- **不改**仓库任何文件（`start.sh`/`stop.sh`/`scripts/*`/`zig/patches/*` 全部原样）。
- **不改**镜像、模型、量化、unit 定义、host 驱动参数（`RmProfilingAdminOnly` 保持 1）。
- profiling 权限通过**临时容器 `--cap-add SYS_ADMIN`** 取得，不改 host。
- 本轮 rank 0 用 `DRY_RUN=1 ./start.sh` 打印的原始 docker 命令**原样复制**，
  仅两处增量：①加 `--cap-add SYS_ADMIN`；②在 CMD 的 `tensorfold-native serve` 前插入 `ncu`。
  rank 1 完全原样。
- 容器名沿用 `dsv41-1017-exl3-tf`，故 `./stop.sh` 能正常收拾；测完仍走 systemd 恢复。

## 5. 计时口径警告

**ncu 下的耗时不可用于速度结论**：kernel replay 会串行化并重复执行内核。
本轮只取**计数器（字节、指令）**，速度仍用既有生产 trace 的 20.162 ms。
## 6. 事故记录：本文件的恢复路径失效了（必读）

**r1 实际结局：head 节点内存被打爆，上述 `systemctl --user start` 恢复路径不可用。**
症状：ICMP 可通（RTT 5–14 ms 抖动大），但 sshd **banner exchange 超时**，
LAN 192.168.1.118 与 CX7 192.168.100.10 两条链路均如此；worker 192.168.1.117 正常
（load 2.07，其 rank 仍 Up）。**由用户人工拔电源恢复。**

**本文件第 4 节的两条自我约束当时没做到**：
- 手动 `docker run` 复制命令 ⇒ `start.sh` 的 `memguard_start` 未运行，失去低内存自动杀 rank 保护。
- 实验容器（rank 0 本体）**没有 `--memory` 硬上限**，且 ncu 的 replay 缓冲无法用容器上限约束
  （它就在 rank 0 进程内）。

**修正后的方法边界**（已写入 `docs/performance/01-发现与结论.md` §0）：
驻留态**禁止** kernel replay 类 profiling。GB10 统一内存 + 富余仅 5–8 GB
⇒ ncu 多 pass 的 save/restore 必然 OOM，**不是调参问题**。
驻留态只允许无重放观测；需要重放必须先停服并把工作集限在个位数 GB。
