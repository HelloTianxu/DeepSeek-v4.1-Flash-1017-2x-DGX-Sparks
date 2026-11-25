#!/usr/bin/env bash
# E-RECLEAN-01 第三步：在「当前配置 + 当前权威负载」下重采 rank0 kernel timeline。
#
# 两重口径污染使得 1123 trace 的份额账不可换算：
#   ① ENGRAM_AIO=1（11-24 已改 0）② prompt 8628 token（当前 10762）
#
# 安全设计：
#   - 先 systemctl --user stop，避免 systemd 把手动起的采集容器判为失败并重启（两个引擎并行会爆内存）
#   - trap EXIT 一律走原 start.sh 恢复生产，并跑 check-boot 核验
#   - 不改生产 start.sh，只用副本 start-trace.sh
#   - docker stop -t 120 给 nsys 足够时间写出报告（SIGTERM 触发）
set -uo pipefail
P=/home/ztxjzh/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks
W=/home/ztxjzh/ptxwork/reclean
SVC=dsv41-tensorfold.service
cd "$P"

restore() {
  echo "=== [trap] 恢复生产 ==="
  docker stop -t 30 dsv41-1017-exl3-tf >/dev/null 2>&1 || true
  ./stop.sh >/dev/null 2>&1 || true
  systemctl --user start "$SVC" || true
  for i in $(seq 1 40); do
    sleep 3
    code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8888/health 2>/dev/null || true)
    [ "$code" = 200 ] && { echo "  health 200 at t=$((i*3))s"; break; }
  done
  echo "  image: $(docker inspect --format '{{.Config.Image}}' dsv41-1017-exl3-tf 2>/dev/null)"
  echo "  avail_GB: $(free -g | awk '/^内存|^Mem/{print $7}')"
}
trap restore EXIT

cp wrap-trace.sh "$W/wrap.sh"
chmod +x "$W/wrap.sh"
echo "=== 停 systemd 托管 ==="
systemctl --user stop "$SVC"
./stop.sh >/dev/null 2>&1 || true
sleep 3
echo "  avail_GB after stop: $(free -g | awk '/^内存|^Mem/{print $7}')"

echo "=== 启动采集版（rank0 带 nsys）==="
# PREPARE=0 必须与生产 systemd 的 Environment=FOREGROUND=1 PREPARE=0 一致：
# 否则 start.sh 会跑 scripts/prepare.sh，其 kit MANIFEST 内容校验在本仓（v0.7.5 严格超集）下失败。
PREPARE=0 ./start-trace.sh restart >"$W/start-trace.log" 2>&1 || { echo "START_FAILED"; tail -20 "$W/start-trace.log"; exit 1; }
for i in $(seq 1 50); do
  sleep 3
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8888/health 2>/dev/null || true)
  [ "$code" = 200 ] && { echo "  health 200 at t=$((i*3))s"; break; }
done
echo "  nsys 在跑: $(docker exec dsv41-1017-exl3-tf pgrep -c nsys 2>/dev/null || echo 0)"

echo "=== 两个请求：r1 预热（冷），r2 热态（用于份额分析）==="
for r in 1 2; do
  python3 "$W/cold_prompt.py" "$W/tr-pf-r$r.json" 28038 >/dev/null 2>&1
  python3 -c "
import json;d=json.load(open('$W/tr-pf-r$r.json'))
print('  r$r tok_s',round(d['server_tok_s_under_ncu_NOT_A_SPEED_CLAIM'],2),'tok',d['uncached_input_tokens'])"
done

echo "=== 停 rank0，触发 nsys 写出（最多 120s）==="
docker stop -t 120 dsv41-1017-exl3-tf >/dev/null 2>&1 || true
ls -la "$W"/trace-current* 2>/dev/null || echo "  NO_REPORT"