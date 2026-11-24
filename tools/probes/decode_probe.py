#!/usr/bin/env python3
"""Decode 实测探针：分离 prefill/decode，核实 MTP 接受率与每轮 token 数。

用法: python3 decode_probe.py <label> [max_tokens] [temp] [reps]
全程单流；请求前查 MemAvailable，低于 4 GiB 直接拒绝。
"""
import json, sys, time, urllib.request, os, statistics

HOST = "http://127.0.0.1:8888"
KEY = open(os.path.expanduser("~/.local/state/dsv41-tensorfold/api-key")).read().strip()

label = sys.argv[1] if len(sys.argv) > 1 else "d"
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 256
TEMP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3


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
        parts = ln.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        try:
            out[parts[0]] = float(parts[1])
        except ValueError:
            pass
    return out


def pick(m, base):
    """取 histogram 的 _sum 与 _count，或裸 counter。"""
    if base + "_sum" in m:
        return m[base + "_sum"], m.get(base + "_count", 0.0)
    return m.get(base, 0.0), 0.0


PROMPT = "Write a detailed technical explanation of how a B-tree index works in a relational database."

rows = []
m0 = memavail()
if m0 < 4.0:
    print(json.dumps({"error": "MemAvailable below 4 GiB", "mem": m0}))
    sys.exit(1)

for i in range(REPS):
    a = metrics()
    body = {
        "model": "DeepSeek-V4.1-Flash-1017",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAXTOK,
        "temperature": TEMP,
        "stream": False,
    }
    req = urllib.request.Request(
        HOST + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"},
    )
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read().decode())
    wall = time.time() - t0
    b = metrics()

    dec_s, dec_n = pick(b, "tensorfold:request_decode_seconds")
    dec_s0, dec_n0 = pick(a, "tensorfold:request_decode_seconds")
    pre_s, _ = pick(b, "tensorfold:request_prefill_seconds")
    pre_s0, _ = pick(a, "tensorfold:request_prefill_seconds")
    lat_s, _ = pick(b, "tensorfold:request_latency_seconds")
    lat_s0, _ = pick(a, "tensorfold:request_latency_seconds")

    d = lambda k: b.get(k, 0.0) - a.get(k, 0.0)
    gen = d("tensorfold:generation_tokens_total")
    prompt = d("tensorfold:prompt_tokens_total")
    rounds = d("tensorfold:decode_rounds_total")
    drafted = d("tensorfold:mtp_drafted_total")
    accepted = d("tensorfold:mtp_accepted_total")
    ds = dec_s - dec_s0
    ps = pre_s - pre_s0

    row = {
        "rep": i + 1,
        "wall_s": round(wall, 6),
        "prompt_tokens": prompt,
        "gen_tokens": gen,
        "prefill_s": round(ps, 6),
        "decode_s": round(ds, 6),
        "latency_s": round(lat_s - lat_s0, 6),
        "decode_tok_s": round(gen / ds, 3) if ds > 0 else None,
        "e2e_tok_s": round(gen / wall, 3),
        "decode_rounds": rounds,
        "tok_per_round": round(gen / rounds, 4) if rounds > 0 else None,
        "mtp_drafted": drafted,
        "mtp_accepted": accepted,
        "accept_rate": round(accepted / drafted, 4) if drafted > 0 else None,
        "mem_after_GiB": round(memavail(), 2),
        "finish": resp["choices"][0].get("finish_reason"),
    }
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False))
    time.sleep(1.0)

ok = [r["decode_tok_s"] for r in rows[1:] if r["decode_tok_s"]] or [
    r["decode_tok_s"] for r in rows if r["decode_tok_s"]
]
summary = {
    "label": label,
    "max_tokens": MAXTOK,
    "temperature": TEMP,
    "reps": REPS,
    "mem_before_GiB": round(m0, 2),
    "decode_tok_s_median_excl_first": round(statistics.median(ok), 3) if ok else None,
    "decode_tok_s_all": [r["decode_tok_s"] for r in rows],
    "tok_per_round_all": [r["tok_per_round"] for r in rows],
    "accept_rate_all": [r["accept_rate"] for r in rows],
    "rows": rows,
}
print("SUMMARY " + json.dumps(summary, ensure_ascii=False))