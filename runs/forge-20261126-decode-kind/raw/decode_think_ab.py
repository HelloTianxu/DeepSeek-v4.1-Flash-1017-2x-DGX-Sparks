#!/usr/bin/env python3
"""thinking 开关对单路 decode 的同场 A/B/A/B 交替判决（code prompt 固定）。

纪律：同场重测对照臂、交替配对看差值是否同向（手册 §7.22/§7.28）。
口径：decode_tok_s = generation_tokens_total / request_decode_seconds_sum 增量。
"""
import json, time, urllib.request, os, statistics

KEY = open(os.path.expanduser("~/.local/state/dsv41-tensorfold/api-key")).read().strip()
H = "http://127.0.0.1:8888"
P = ("Write a complete Python class implementing a red-black tree with insert, delete, "
     "search and rebalancing. Include type hints and a docstring for every method.")


def met():
    r = urllib.request.Request(H + "/metrics", headers={"Authorization": "Bearer " + KEY})
    o = {}
    for ln in urllib.request.urlopen(r, timeout=10).read().decode().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.rsplit(" ", 1)
        if len(p) == 2:
            try:
                o[p[0]] = float(p[1])
            except ValueError:
                pass
    return o


def run(think):
    a = met()
    b = {"model": "DeepSeek-V4.1-Flash-1017",
         "messages": [{"role": "user", "content": P}],
         "max_tokens": 256, "temperature": 0.0, "stream": False,
         "chat_template_kwargs": {"enable_thinking": think}}
    q = urllib.request.Request(H + "/v1/chat/completions", data=json.dumps(b).encode(),
                               headers={"Authorization": "Bearer " + KEY,
                                        "Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(q, timeout=300).read().decode())
    c = met()
    g = lambda k: c.get(k, 0.0) - a.get(k, 0.0)
    ds = g("tensorfold:request_decode_seconds_sum")
    gen = g("tensorfold:generation_tokens_total")
    rd = g("tensorfold:decode_rounds_total")
    ac = g("tensorfold:mtp_accepted_total")
    dr = g("tensorfold:mtp_drafted_total")
    rtok = (resp.get("usage", {}).get("completion_tokens_details", {}) or {}).get("reasoning_tokens", 0)
    return gen / ds, ac / dr, gen / rd, rtok


print("code prompt 同场 A/B/A/B 交替 (on=thinking ON, off=OFF)")
res = {"on": [], "off": []}
rows = []
for i in range(3):
    for t in (True, False):
        v, a, r, rt = run(t)
        k = "on" if t else "off"
        res[k].append(v)
        rows.append({"pair": i + 1, "arm": k, "tok_s": round(v, 3),
                     "accept": round(a, 4), "tok_per_round": round(r, 4), "reasoning_tokens": rt})
        print("  pair%d %-3s %7.3f tok/s  accept=%.4f tok/round=%.4f rtok=%d" % (i + 1, k, v, a, r, rt))
        time.sleep(1.0)

ds = [res["off"][i] - res["on"][i] for i in range(3)]
print("配对差 (off-on):")
for i, d in enumerate(ds):
    print("  pair%d: %+.3f tok/s" % (i + 1, d))
mon = statistics.median(res["on"])
moff = statistics.median(res["off"])
verdict = {
    "on_all": [round(x, 3) for x in res["on"]],
    "off_all": [round(x, 3) for x in res["off"]],
    "pair_diffs": [round(d, 3) for d in ds],
    "all_same_direction": all(d > 0 for d in ds),
    "median_diff": round(statistics.median(ds), 3),
    "on_median": round(mon, 3), "off_median": round(moff, 3),
    "gain_pct": round((moff / mon - 1) * 100, 2),
    "rows": rows,
}
print("同向: %s  中位差 %+.3f" % (verdict["all_same_direction"], verdict["median_diff"]))
print("on 中位 %.3f  off 中位 %.3f  增益 %+.2f%%" % (mon, moff, verdict["gain_pct"]))
print("VERDICT " + json.dumps(verdict, ensure_ascii=False))