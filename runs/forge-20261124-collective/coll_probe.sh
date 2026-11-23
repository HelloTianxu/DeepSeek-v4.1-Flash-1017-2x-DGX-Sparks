#!/usr/bin/env bash
# collective 归因：冷前缀请求期间高频采样两节点 RDMA 计数器。
# 全程只读 sysfs + 一次正常推理请求；不注入引擎、不改配置、不重启。
# 用法：bash coll_probe.sh <label> [target_chars] [sample_seconds]
set -euo pipefail

LABEL="${1:?label required}"
CHARS="${2:-28000}"
DUR="${3:-30}"
WORKER=ztxjzh@192.168.100.11
OUT=/tmp/coll-$LABEL
MIN_GIB=5.0

mkdir -p "$OUT"

chk() { awk '/^MemAvailable:/{printf "%.3f", $2/1048576}' /proc/meminfo; }
H=$(chk)
W=$(ssh -o BatchMode=yes "$WORKER" "awk '/^MemAvailable:/{printf \"%.3f\", \$2/1048576}' /proc/meminfo")
echo "[$LABEL] MemAvailable head=$H GiB worker=$W GiB"
awk -v a="$H" -v b="$MIN_GIB" 'BEGIN{exit !(a<b)}' && { echo "ABORT: head 内存不足"; exit 1; }
awk -v a="$W" -v b="$MIN_GIB" 'BEGIN{exit !(a<b)}' && { echo "ABORT: worker 内存不足"; exit 1; }

# 记录配置快照（MTU / active_mtu），供 A/B 对照
{
  echo "label=$LABEL chars=$CHARS"
  echo "head_mem=$H worker_mem=$W"
  for i in enp1s0f1np1 enP2p1s0f1np1; do
    echo "head_mtu_$i=$(cat /sys/class/net/$i/mtu 2>/dev/null)"
  done
  for d in rocep1s0f1 roceP2p1s0f1; do
    echo "head_${d}_rate=$(cat /sys/class/infiniband/$d/ports/1/rate 2>/dev/null)"
  done
  ssh -o BatchMode=yes "$WORKER" 'for i in enp1s0f1np1 enP2p1s0f1np1; do
      echo "worker_mtu_$i=$(cat /sys/class/net/$i/mtu 2>/dev/null)"; done'
} > "$OUT/config.txt"

# 两节点同时起采样器
ssh -o BatchMode=yes "$WORKER" \
  "nohup python3 /tmp/rdma_sample.py /tmp/coll-$LABEL-worker.csv $DUR 10 >/tmp/coll-$LABEL-worker.log 2>&1 &" &
python3 /tmp/rdma_sample.py "$OUT/head.csv" "$DUR" 10 > "$OUT/head-sampler.log" 2>&1 &
SPID=$!
sleep 2   # 让采样器先跑起来，建立空闲基线

# 冷前缀请求，记录墙钟窗口
python3 - "$OUT/request.json" "$CHARS" <<'PY'
import json, subprocess, sys, time, pathlib, urllib.request
out, chars = sys.argv[1], sys.argv[2]
BASE = "http://127.0.0.1:8888"
KEY = pathlib.Path("/home/ztxjzh/.local/state/dsv41-tensorfold/api-key").read_text().strip()

def snap():
    r = urllib.request.Request(BASE + "/metrics", headers={"Authorization": f"Bearer {KEY}"})
    m = {}
    for line in urllib.request.urlopen(r, timeout=10).read().decode().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, val = line.rpartition(" ")
        try:
            m[name.split("{")[0]] = m.get(name.split("{")[0], 0.0) + float(val)
        except ValueError:
            pass
    return m

m0 = snap()
t0 = time.time()
r = subprocess.run([sys.executable, "/tmp/cold_prompt.py", "/tmp/_cp_inner.json", chars],
                   capture_output=True, text=True)
t1 = time.time()
m1 = snap()
d = {}
try:
    d = json.load(open("/tmp/_cp_inner.json"))
except Exception as e:
    d = {"error": str(e)}
d["probe_wall_start"] = t0
d["probe_wall_end"] = t1
d["probe_wall_seconds"] = t1 - t0
# 只保留发生变化的指标，用于把 KV shard 等非 collective 流量分离出去
d["metrics_delta"] = {k: m1[k] - m0.get(k, 0.0) for k in m1
                      if abs(m1[k] - m0.get(k, 0.0)) > 0}
d["stdout_tail"] = r.stdout[-400:]
json.dump(d, open(out, "w"), ensure_ascii=False, indent=2)
print(f"request wall {t1-t0:.3f}s")
PY

wait $SPID 2>/dev/null || true
sleep 1
scp -q -o BatchMode=yes "$WORKER:/tmp/coll-$LABEL-worker.csv" "$OUT/worker.csv"
cat "$OUT/head-sampler.log"
echo "[$LABEL] 完成，产物在 $OUT"
awk '/^MemAvailable:/{printf "结束后 head avail %.2f GiB\n", $2/1048576}' /proc/meminfo