#!/usr/bin/env python3
"""E-DRAM-01：发一个冷前缀长 prompt，触发真实专家内核供 ncu 取计数器。

口径沿用既有纪律：主报 (prompt_tokens_total - prompt_tokens_cached_total) / request_prefill_time_seconds。
nonce 置于位置 0 保证前缀缓存命中 0；corpus 重复 8 份，长度对齐既有 8628 token 量级。
max_tokens=1，避免 decode 混入。

注意：本请求在 ncu 下运行，**其耗时不可用于速度结论**（kernel replay 串行化）。
速度仍引用既有生产 trace。本脚本的 tok/s 仅作"请求确实完成"的存在性证据。

用法：python3 cold_prompt.py <out.json> [target_chars]
"""
import json, sys, time, urllib.request, pathlib, hashlib

BASE = "http://127.0.0.1:8888"
KEY = pathlib.Path("/home/ztxjzh/.local/state/dsv41-tensorfold/api-key").read_text().strip()
HDR = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


def metrics():
    req = urllib.request.Request(BASE + "/metrics", headers=HDR)
    out = {}
    for line in urllib.request.urlopen(req, timeout=10).read().decode().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, val = line.rpartition(" ")
        try:
            out[name] = float(val)
        except ValueError:
            pass
    return out


def pick(m, *cands):
    """指标名可能带 label，按前缀聚合。"""
    tot = 0.0
    found = []
    for k, v in m.items():
        base = k.split("{")[0]
        if base in cands:
            tot += v
            found.append(k)
    return tot, found


# 混合语料：代码 + 自然语言，两类都覆盖（既有纪律要求）
BLOCK = (
    "def reconcile_ledger(rows, tolerance=1e-6):\n"
    "    total = 0.0\n"
    "    for r in rows:\n"
    "        if r['kind'] == 'expert':\n"
    "            total += r['bytes'] * r['weight']\n"
    "    return total\n"
    "# The grouped matmul reads each expert's quantized panels once per tile.\n"
    "class BandwidthAccount:\n"
    "    def __init__(self, peak):\n"
    "        self.peak = peak\n"
    "        self.samples = []\n"
    "    def add(self, nbytes, seconds):\n"
    "        self.samples.append(nbytes / seconds)\n"
    "    def median(self):\n"
    "        s = sorted(self.samples)\n"
    "        return s[len(s) // 2]\n"
    "在带宽受限的推理阶段，唯一字节数除以耗时得到的是有效带宽，"
    "它只能作为下界，因为实际流量可能包含重复读取。"
    "屋顶线模型把算术强度与机器平衡点比较，判断该内核落在算力侧还是带宽侧。"
    "若算术强度低于平衡点，则提高算力不会缩短耗时，必须减少搬运字节或提高带宽利用率。"
    "专家混合模型的前填充阶段每个 token 只激活少量专家，但一个 chunk 内所有专家几乎都会被触达，"
    "因此权重读取总量近似等于全部专家权重，与 chunk 内的 token 数关系很弱。"
    "这解释了为什么增大 chunk 能摊薄每 token 的权重读取成本。"
    "量化把权重压到更少的比特，但解码指令与张量计算仍以半精度进行，"
    "所以压缩存储不等于按比例缩短时间，必须实测同一形状下的可持续吞吐。"
) * 2


def main():
    out = pathlib.Path(sys.argv[1])
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 28000
    nonce = hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:16]
    body = (BLOCK * 64)[:target]
    prompt = f"nonce-{nonce}\n" + body + "\nReply with OK."

    m0 = metrics()
    payload = {
        "model": "DeepSeek-V4.1-Flash-1017",
        "max_tokens": 1,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": prompt}],
    }
    t0 = time.time()
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                data=json.dumps(payload).encode(), headers=HDR)
    resp = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    wall = time.time() - t0
    m1 = metrics()

    pt0, _ = pick(m0, "tensorfold:prompt_tokens_total")
    pt1, _ = pick(m1, "tensorfold:prompt_tokens_total")
    pc0, _ = pick(m0, "tensorfold:prompt_tokens_cached_total")
    pc1, _ = pick(m1, "tensorfold:prompt_tokens_cached_total")
    pf0, k0 = pick(m0, "tensorfold:request_prefill_time_seconds_sum")
    pf1, k1 = pick(m1, "tensorfold:request_prefill_time_seconds_sum")
    n0, _ = pick(m0, "tensorfold:request_prefill_time_seconds_count")
    n1, _ = pick(m1, "tensorfold:request_prefill_time_seconds_count")

    uncached = (pt1 - pt0) - (pc1 - pc0)
    dsec = pf1 - pf0
    rec = {
        "nonce": nonce,
        "prompt_chars": len(prompt),
        "usage": resp.get("usage"),
        "prompt_tokens_total_delta": pt1 - pt0,
        "prompt_tokens_cached_delta": pc1 - pc0,
        "uncached_input_tokens": uncached,
        "prefill_seconds_delta": dsec,
        "prefill_count_delta": n1 - n0,
        "prefill_metric_keys": sorted(set(k0 + k1)),
        "server_tok_s_under_ncu_NOT_A_SPEED_CLAIM": (uncached / dsec) if dsec > 0 else None,
        "client_wall_s_under_ncu": wall,
        "reply": (resp["choices"][0]["message"].get("content") or "")[:40],
    }
    out.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    print(json.dumps(rec, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()