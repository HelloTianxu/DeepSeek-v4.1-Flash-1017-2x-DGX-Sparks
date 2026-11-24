#!/usr/bin/env python3
"""E-SELF-05: attribute the chunk-boundary no-kernel stalls. Zero injection.

Reads only /proc: per-thread delayacct_blkio_ticks (field 42), block-I/O byte counters,
and samples thread state + wchan while a cold-prefix probe runs. No ncu, no nsys,
no engine change, no restart.
"""
import json, os, subprocess, sys, threading, time
from collections import Counter, defaultdict

PID = int(sys.argv[1])
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/stall-attr.json"
PROBE = sys.argv[3] if len(sys.argv) > 3 else None
PROBE_ARGS = sys.argv[4:]
HZ = os.sysconf("SC_CLK_TCK")
TASK = f"/proc/{PID}/task"


def threads():
    try:
        return os.listdir(TASK)
    except FileNotFoundError:
        return []


def snap():
    """per-thread blkio ticks + utime/stime, and process io counters."""
    per = {}
    for t in threads():
        try:
            with open(f"{TASK}/{t}/stat") as f:
                raw = f.read()
            # comm may contain spaces/parens -> split after the last ')'
            rp = raw.rindex(")")
            f2 = raw[rp + 2:].split()
            # f2[0] is state (field 3); field N -> f2[N-3]
            per[t] = {
                "blkio": int(f2[42 - 3]),
                "utime": int(f2[14 - 3]),
                "stime": int(f2[15 - 3]),
                "state": f2[0],
            }
        except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
            pass
    io = {}
    try:
        with open(f"/proc/{PID}/io") as f:
            for ln in f:
                k, v = ln.split(":", 1)
                io[k.strip()] = int(v.strip())
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return per, io


stop = threading.Event()
samples = []          # (t_ms, n_R, n_S, n_D, Counter(wchan of D))
dstate_detail = []


def sampler():
    t0 = time.monotonic()
    while not stop.is_set():
        nR = nS = nD = nOther = 0
        wc = Counter()
        allw = Counter()
        for t in threads():
            try:
                with open(f"{TASK}/{t}/stat") as f:
                    raw = f.read()
                st = raw[raw.rindex(")") + 2]
                if st == "R":
                    nR += 1
                    allw["R:running"] += 1
                    continue
                if st == "D":
                    nD += 1
                elif st == "S":
                    nS += 1
                else:
                    nOther += 1
                try:
                    with open(f"{TASK}/{t}/wchan") as g:
                        w = g.read().strip() or "?"
                except OSError:
                    w = "?"
                allw[f"{st}:{w}"] += 1
                if st == "D":
                    wc[w] += 1
                    dstate_detail.append((round((time.monotonic() - t0) * 1e3, 1), t, w))
            except (FileNotFoundError, ProcessLookupError, ValueError):
                pass
        samples.append((round((time.monotonic() - t0) * 1e3, 1), nR, nS, nD, dict(wc), dict(allw)))
        stop.wait(0.020)


a_per, a_io = snap()
th = threading.Thread(target=sampler, daemon=True)
th.start()
t_start = time.monotonic()

probe_out, probe_rc = None, None
if PROBE:
    p = subprocess.run([sys.executable, PROBE] + PROBE_ARGS, capture_output=True, text=True)
    probe_rc = p.returncode
    try:
        probe_out = json.loads(p.stdout)
    except Exception:
        probe_out = {"stdout": p.stdout[-3000:], "stderr": p.stderr[-2000:]}

wall = time.monotonic() - t_start
stop.set()
th.join(timeout=2)
b_per, b_io = snap()

# deltas
blk = []
for t, b in b_per.items():
    a = a_per.get(t)
    if not a:
        continue
    d = b["blkio"] - a["blkio"]
    if d > 0:
        blk.append((t, d, round(d / HZ * 1e3, 1)))
blk.sort(key=lambda x: -x[1])
tot_blk_ticks = sum(x[1] for x in blk)

io_delta = {k: b_io.get(k, 0) - a_io.get(k, 0) for k in set(a_io) | set(b_io)}

# window where D-state threads were seen
res = {
    "pid": PID,
    "hz": HZ,
    "wall_s": round(wall, 6),
    "probe_rc": probe_rc,
    "probe": probe_out,
    "blkio_total_ms_all_threads": round(tot_blk_ticks / HZ * 1e3, 1),
    "blkio_top_threads": [{"tid": t, "ticks": d, "ms": ms} for t, d, ms in blk[:15]],
    "io_delta": io_delta,
    "n_samples": len(samples),
    "samples_with_D": sum(1 for s in samples if s[3] > 0),
    "d_wchan_totals": dict(sum((Counter(s[4]) for s in samples), Counter())),
    "all_wchan_totals": dict(sum((Counter(s[5]) for s in samples if len(s) > 5), Counter())),
    "d_episodes": dstate_detail[:400],
    "timeline": samples,
}
with open(OUT, "w") as f:
    json.dump(res, f)

print(f"wall            {wall:.3f} s")
if probe_out and isinstance(probe_out, dict):
    for k in ("uncached_input_tokens", "prefill_seconds_delta", "prefill_tok_s", "server_tok_s"):
        if k in probe_out:
            print(f"{k:16}{probe_out[k]}")
print(f"blkio wait      {tot_blk_ticks / HZ * 1e3:.1f} ms  (all threads summed)")
print(f"io_delta        read_bytes={io_delta.get('read_bytes', 0) / 1e6:.1f} MB  "
      f"rchar={io_delta.get('rchar', 0) / 1e6:.1f} MB")
print(f"D-state samples {res['samples_with_D']}/{len(samples)}")
print("D wchan:", res["d_wchan_totals"])
print("top blkio threads:", res["blkio_top_threads"][:6])
print("written", OUT)