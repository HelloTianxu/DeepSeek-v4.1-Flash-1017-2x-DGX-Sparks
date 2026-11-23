# 停服测试的恢复方式（测试前先落盘）

本轮经用户明确授权停服测试（"这个TP2环境你可以停服测试，现在没有在使用。注意，只要不破坏推理环境就可以"）。

## 停服前快照

- head：`spark-bada` = 192.168.1.118（mDNS `spark-bada.local`），ssh 用户 `ztxjzh`
- worker：`spark-bbf1` = 192.168.1.117
- 容器：`dsv41-1017-exl3-tf`，镜像 `tensorfold-dsv41:exl3-tp2-8348338a8d90`，停服前 Up 8 hours
- 服务单元：**user systemd** `dsv41-tensorfold.service`，active，NRestarts=0
  - WorkingDirectory `/home/ztxjzh/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks`
  - ExecStart `/usr/bin/bash ./start.sh`；ExecStop `/usr/bin/bash ./stop.sh`
  - Restart=on-failure，RestartSec=60s，**StartLimitBurst=3 / 30min**
- API `http://127.0.0.1:8888/health` 停服前 `{"status": "ok"}`
- GPU 空载：SM 2333 MHz / 上限 3003 MHz，48C，10.11 W，util 0%
- 停服前显存：torch `mem_get_info` free=0.8GiB / total=121.6GiB（模型已驻留）

## 恢复命令（唯一路径，勿直接跑 start.sh）

```bash
ssh ztxjzh@192.168.1.118 'systemctl --user start dsv41-tensorfold.service'
# 若触发 StartLimitBurst=3/30min：
ssh ztxjzh@192.168.1.118 'systemctl --user reset-failed dsv41-tensorfold.service && systemctl --user start dsv41-tensorfold.service'
```

**不改 unit 限制、不改配置/镜像/模型、不杀外部进程、不删锁。** 停服只用
`systemctl --user stop dsv41-tensorfold.service`（其 ExecStop 即项目 stop.sh，会一并停 worker）。

## 验收

恢复后必须 `curl -s http://127.0.0.1:8888/health` 返回 ok，且容器 Up、两 rank 正常。
