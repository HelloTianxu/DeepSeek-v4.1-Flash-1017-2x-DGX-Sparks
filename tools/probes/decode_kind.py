#!/usr/bin/env python3
"""decode 内容分型对照：同引擎同时刻，只变 prompt 内容，其余全部固定。

回答「单路是不是从 100 掉到 60」。假设：60 是短问答/random 档，
100+ 是 code 档，差异来自 MTP 接受率（内容可预测性），不是引擎回退。

口径完全复用 tools/probes/decode_probe.py：decode_tok_s = generation_tokens / request_decode_seconds，
已分离 prefill。四档 prompt 长度刻意接近，以控制 KV 长度这个混淆变量。

用法: python3 decode_kind.py [max_tokens] [reps]
"""
import json, sys, time, urllib.request, os, statistics, random

HOST = "http://127.0.0.1:8888"
KEY = open(os.path.expanduser("~/.local/state/dsv41-tensorfold/api-key")).read().strip()
MAXTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 256
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
# thinking 默认 OFF，与 tools/bench.py 权威口径一致（bench.py:115 chat_template_kwargs）。
# 关键：decode_probe.py 未关 thinking，其 59.265 基线是 thinking ON 口径，两者不可比。
THINK = (len(sys.argv) > 3 and sys.argv[3].lower() == "on")

random.seed(1017)
WORDS = ["ledger", "quartz", "meadow", "fractal", "cobalt", "thistle", "harbor", "vellum",
         "cinder", "plinth", "orchid", "basalt", "wicker", "murmur", "gannet", "sable"]
RAND = "Continue this text with more of the same: " + " ".join(random.choice(WORDS) for _ in range(28))

# 四档：长度相近的短指令，只有「要求生成的内容类型」不同
KINDS = {
    "qa":    "Write a detailed technical explanation of how a B-tree index works in a relational database.",
    "code":  "Write a complete Python class implementing a red-black tree with insert, delete, search and rebalancing. Include type hints and a docstring for every method.",
    "prose": "Write a flowing descriptive essay about the experience of walking through an old city at dawn, with rich sensory detail and no lists.",
    "rand":  RAND,
    "boiler": "Write 20 Python dataclasses for a REST API: User, Order, Product, Invoice, Payment, Shipment, Address, Coupon, Review, Cart, CartItem, Wishlist, Category, Tag, Supplier, Warehouse, Inventory, Refund, Subscription, Notification. Each with 6 typed fields and a docstring.",
}


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
    if base + "_sum" in m:
        return m[base + "_sum"], m.get(base + "_count", 0.0)
    return m.get(base, 0.0), 0.0


if memavail() < 4.0:
    print(json.dumps({"error": "MemAvailable below 4 GiB"}))
    sys.exit(1)

result = {"_thinking": THINK}
for kind, prompt in KINDS.items():
    rows = []
    for i in range(REPS):
        a = metrics()
        body = {"model": "DeepSeek-V4.1-Flash-1017",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": MAXTOK, "temperature": 0.0, "stream": False,
                "chat_template_kwargs": {"enable_thinking": THINK}}
        req = urllib.request.Request(HOST + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Authorization": "Bearer " + KEY,
                                              "Content-Type": "application/json"})
        t0 = time.time()
        resp = json.loads(urllib.request.urlopen(req, timeout=600).read().decode())
        wall = time.time() - t0
        b = metrics()
        d = lambda k: b.get(k, 0.0) - a.get(k, 0.0)
        ds = pick(b, "tensorfold:request_decode_seconds")[0] - pick(a, "tensorfold:request_decode_seconds")[0]
        ps = pick(b, "tensorfold:request_prefill_seconds")[0] - pick(a, "tensorfold:request_prefill_seconds")[0]
        gen = d("tensorfold:generation_tokens_total")
        rounds = d("tensorfold:decode_rounds_total")
        drafted = d("tensorfold:mtp_drafted_total")
        accepted = d("tensorfold:mtp_accepted_total")
        msg = resp["choices"][0]["message"]
        txt = msg.get("content") or ""
        rtok = (resp.get("usage", {}).get("completion_tokens_details", {}) or {}).get("reasoning_tokens", 0)
        rows.append({
            "rep": i + 1, "prompt_tokens": d("tensorfold:prompt_tokens_total"),
            "gen_tokens": gen, "prefill_s": round(ps, 4), "decode_s": round(ds, 4),
            "decode_tok_s": round(gen / ds, 3) if ds > 0 else None,
            "decode_rounds": rounds,
            "tok_per_round": round(gen / rounds, 4) if rounds > 0 else None,
            "accept_rate": round(accepted / drafted, 4) if drafted > 0 else None,
            "finish": resp["choices"][0].get("finish_reason"),
            "reasoning_tokens": rtok, "content_len": len(txt),
            "sample": txt[:100].replace("\n", "\\n"),
        })
        time.sleep(1.0)
    ok = [r["decode_tok_s"] for r in rows[1:] if r["decode_tok_s"]] or \
         [r["decode_tok_s"] for r in rows if r["decode_tok_s"]]
    med = round(statistics.median(ok), 3) if ok else None
    acc = [r["accept_rate"] for r in rows if r["accept_rate"]]
    tpr = [r["tok_per_round"] for r in rows if r["tok_per_round"]]
    result[kind] = {
        "decode_tok_s_median_excl_first": med,
        "decode_tok_s_all": [r["decode_tok_s"] for r in rows],
        "accept_rate_median": round(statistics.median(acc), 4) if acc else None,
        "tok_per_round_median": round(statistics.median(tpr), 4) if tpr else None,
        "prompt_tokens": rows[0]["prompt_tokens"],
        "sample": rows[-1]["sample"], "rows": rows,
    }
    print(f"{kind:6s} tok/s={med}  accept={result[kind]['accept_rate_median']}  "
          f"tok/round={result[kind]['tok_per_round_median']}  ptok={rows[0]['prompt_tokens']}  "
          f"rtok={rows[-1]['reasoning_tokens']}  clen={rows[-1]['content_len']}")

print("SUMMARY " + json.dumps(result, ensure_ascii=False))