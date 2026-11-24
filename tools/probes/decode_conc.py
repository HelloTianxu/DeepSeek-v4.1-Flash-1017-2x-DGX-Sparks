#!/usr/bin/env python3
"""Decode 并发探针：N 路同时单流生成，测聚合吞吐与每路速度。

用法: python3 decode_conc.py <N> [max_tokens] [temp] [reps]
内存闸：MemAvailable < 4 GiB 拒绝。
"""
import json, sys, time, urllib.request, os, statistics
from concurrent.futures import ThreadPoolExecutor

HOST = "http://127.0.0.1:8888"
KEY = open(os.path.expanduser("~/.local/state/dsv41-tensorfold/api-key")).read().strip()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 256
TEMP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 2

PROMPTS = [
    "Write a detailed technical explanation of how a B-tree index works in a relational database.",
    "Explain the differences between TCP and UDP, including congestion control and use cases.",
    "Describe how a modern out-of-order CPU pipeline handles branch misprediction.",
    "Explain how log-structured merge trees compact data and why write amplification matters.",
    "Describe the CAP theorem and give concrete examples of systems on each side.",
    "Explain how TLS 1.3 establishes a session with forward secrecy.",
    "Describe how garbage collection works in a generational heap.",
    "Explain vector clocks and how they detect concurrent updates.",
]


def memavail():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) / 1048576
    return 0.0


def metrics():
    r = urllib.request.Request(HOST + "/metrics", headers={"Authorization": "Bearer " + KEY})
    out = {}
    for ln in urllib.request.urlopen(r, timeout=10).read().decode().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.rsplit(" ", 1)
        if len(p) == 2:
            try:
                out[p[0]] = float(p[1])
            except ValueError:
                pass
    return out


def one(i):
    body = {
        "model": "DeepSeek-V4.1-Flash-1017",
        "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
        "max_tokens": MAXTOK, "temperature": TEMP, "stream": False,
    }
    req = urllib.request.Request(
        HOST + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=900).read().decode())
    w = time.time() - t0
    u = r.get("usage", {})
    return {"i": i, "wall_s": w, "gen": u.get("completion_tokens"), "prompt": u.get("prompt_tokens")}


if memavail() < 4.0:
    print(json.dumps({"error": "mem below 4 GiB"})); sys.exit(1)

rows = []
for rep in range(REPS):
    a = metrics()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N) as ex:
        res = list(ex.map(one, range(N)))
    wall = time.time() - t0
    b = metrics()
    d = lambda k: b.get(k, 0.0) - a.get(k, 0.0)
    gen = d("tensorfold:generation_tokens_total")
    rounds = d("tensorfold:decode_rounds_total")
    dr = d("tensorfold:mtp_drafted_total")
    ac = d("tensorfold:mtp_accepted_total")
    ds = d("tensorfold:request_decode_seconds_sum")
    row = {
        "rep": rep + 1, "N": N, "wall_s": round(wall, 4),
        "gen_total": gen, "agg_tok_s": round(gen / wall, 3),
        "per_stream_tok_s": round(gen / wall / N, 3),
        "decode_rounds": rounds,
        "tok_per_round": round(gen / rounds, 4) if rounds else None,
        "rounds_per_s": round(rounds / wall, 3),
        "accept_rate": round(ac / dr, 4) if dr else None,
        "sum_decode_s": round(ds, 4),
        "mem_GiB": round(memavail(), 2),
        "per_req_wall": [round(r["wall_s"], 3) for r in res],
    }
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False))
    time.sleep(2.0)

use = rows[1:] or rows
print("SUMMARY " + json.dumps({
    "N": N, "max_tokens": MAXTOK, "temperature": TEMP,
    "agg_tok_s_median": round(statistics.median([r["agg_tok_s"] for r in use]), 3),
    "per_stream_median": round(statistics.median([r["per_stream_tok_s"] for r in use]), 3),
    "rows": rows}, ensure_ascii=False))