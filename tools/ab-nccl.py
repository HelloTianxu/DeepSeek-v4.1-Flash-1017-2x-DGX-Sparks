#!/usr/bin/env python3
"""E-COMM-01：NCCL 协议/算法 A/B。每臂改 scripts/local.sh 的 export 块 -> 重启 -> 等就绪 -> 同场 bench。

纪律：
  - 对照臂必须同场重测，禁止复用历史基线（--arms 里显式带 default，且可 A/B/A/B 交替）
  - 就绪判据：/health 200 且 avail <= 12 GiB（刚 start 时 avail 28 GiB 会误报）
  - 臂身份：从 journal 抓 NCCL 自报的 "set by environment" 行；协议是否真被采纳另用内核名验
  - 任何一臂起不来 -> 立即恢复原 local.sh 并重启，不留实验配置在生产
  - decode 必须一起测：AllGather 的小消息走 LL 是为低延迟，强制 Simple 可能拖慢 decode
"""
import argparse, json, os, re, subprocess, sys, time, urllib.request

PROJ = os.path.expanduser("~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks")
LOCAL = os.path.join(PROJ, "scripts/local.sh")
BEGIN = "# >>> E-COMM-01 NCCL A/B (managed block, auto-removed on restore) >>>"
END = "# <<< E-COMM-01 NCCL A/B <<<"

ap = argparse.ArgumentParser()
ap.add_argument("--arms", default="default,Simple,LL128",
                help="逗号分隔；default=不设 NCCL_PROTO；其余可写 PROTO 或 PROTO@ALGO，如 Simple@Ring")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--prefill-sizes", default="8192,32768")
ap.add_argument("--streams", default="1,4")
ap.add_argument("--suite", default="prefill,decode,multi")
ap.add_argument("--outdir", default=os.path.expanduser("~/ptxwork/comm-ab"))
ap.add_argument("--ready-timeout", type=int, default=240)
ap.add_argument("--debug", action="store_true", help="所有臂对称开 NCCL_DEBUG=INFO 以取臂身份自报行")
a = ap.parse_args()

os.makedirs(a.outdir, exist_ok=True)
BACKUP = os.path.join(a.outdir, "local.sh.pre-comm-ab")


def sh(cmd, timeout=600):
    return subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout)


def avail_gib():
    r = sh("free -g | awk '/Mem|内存/{print $7}'")
    try:
        return int(r.stdout.strip().split()[0])
    except Exception:
        return -1


def health():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8888/health", timeout=5) as f:
            return f.status
    except Exception:
        return 0


def strip_block(text):
    return re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", "", text, flags=re.S)


def set_arm(arm):
    base = strip_block(open(LOCAL).read())
    lines = [BEGIN]
    if arm != "default":
        proto, _, algo = arm.partition("@")
        lines.append(f"export NCCL_PROTO={proto}")
        if algo:
            lines.append(f"export NCCL_ALGO={algo}")
    if a.debug:
        lines.append("export NCCL_DEBUG=INFO")
        lines.append("export NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING")
    lines.append(END)
    open(LOCAL, "w").write(base + "\n".join(lines) + "\n")


def restart_ready(tag):
    sh("systemctl --user stop dsv41-tensorfold.service", timeout=180)
    time.sleep(8)
    sh("systemctl --user start dsv41-tensorfold.service", timeout=180)
    t0 = time.time()
    while time.time() - t0 < a.ready_timeout:
        h, av = health(), avail_gib()
        if h == 200 and 0 <= av <= 12:
            print(f"    READY {tag}  {time.time()-t0:.0f}s  avail={av}GiB", flush=True)
            return True
        time.sleep(6)
    print(f"    FAIL-READY {tag} health={health()} avail={avail_gib()}", flush=True)
    return False


def identity(tag):
    r = sh("journalctl --user -u dsv41-tensorfold.service --since '-4min' --no-pager | "
           "grep -iE 'NCCL_(PROTO|ALGO|DEBUG).*set by environment|Protocol|nChannels' | tail -25")
    p = os.path.join(a.outdir, f"nccl-{tag}.txt")
    open(p, "w").write(r.stdout)
    keep = [l for l in r.stdout.splitlines() if "set by environment" in l]
    for l in keep[:6]:
        print("    id: " + l.strip()[-150:], flush=True)
    return keep


if not os.path.exists(BACKUP):
    open(BACKUP, "w").write(open(LOCAL).read())
    print(f"BACKUP {BACKUP}", flush=True)

arms = a.arms.split(",")
results = []
ok = True
try:
    for i, arm in enumerate(arms):
        tag = f"{i+1:02d}-{arm.replace('@','_')}"
        print(f"[{i+1}/{len(arms)}] arm={arm}", flush=True)
        set_arm(arm)
        if not restart_ready(tag):
            ok = False
            break
        ident = identity(tag)
        out = os.path.join(a.outdir, f"bench-{tag}.json")
        cmd = (f"cd {PROJ} && python3 -I tools/bench.py --suite {a.suite} "
               f"--prefill-sizes {a.prefill_sizes} --streams {a.streams} --reps {a.reps} "
               f"--label {tag} --out {out}")
        t0 = time.time()
        r = sh(cmd, timeout=2400)
        open(os.path.join(a.outdir, f"bench-{tag}.log"), "w").write(r.stdout + "\n--stderr--\n" + r.stderr)
        print(f"    bench rc={r.returncode} {time.time()-t0:.0f}s", flush=True)
        if r.returncode != 0:
            print("    BENCH-FAIL, 尾部：\n" + "\n".join(r.stdout.splitlines()[-8:]), flush=True)
            ok = False
            break
        s = json.load(open(out)).get("summary", {})
        row = {"arm": arm, "tag": tag, "id_lines": ident}
        for k, v in s.items():
            if k.startswith(("prefill.", "decode.", "multi.")) and isinstance(v, (int, float)):
                row[k] = v
        results.append(row)
        for k in sorted(row):
            if k.startswith(("prefill.", "decode.", "multi.")):
                print(f"      {k} = {row[k]:.3f}", flush=True)
finally:
    print("恢复原 local.sh 并重启", flush=True)
    open(LOCAL, "w").write(open(BACKUP).read())
    restart_ready("restore")

json.dump(results, open(os.path.join(a.outdir, "summary.json"), "w"), indent=2)
print("\n== 汇总（同场）==", flush=True)
keys = sorted({k for r in results for k in r if k.startswith(("prefill.", "decode.", "multi."))})
print(f"{'metric':<34}" + "".join(f"{r['arm']:>14}" for r in results), flush=True)
for k in keys:
    print(f"{k:<34}" + "".join(f"{r.get(k, float('nan')):>14.3f}" for r in results), flush=True)
print("ALL_OK" if ok else "ABORTED_RESTORED", flush=True)