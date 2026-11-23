import json, struct, math, collections

M = "/home/ztxjzh/.cache/huggingface/tensorfold-dsv41/aee640e2-ablit-0bdc2036/model"
idx = json.load(open(M + "/model.safetensors.index.json"))["weight_map"]

hdr = {}
def head(sh):
    if sh not in hdr:
        with open(M + "/" + sh, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr[sh] = json.loads(f.read(n))
    return hdr[sh]

print("=== embed / lm_head（非 visual）===")
for k in sorted(idx):
    if ("embed_tokens" in k or "lm_head" in k) and "visual" not in k:
        h = head(idx[k])[k]
        sz = h["data_offsets"][1] - h["data_offsets"][0]
        print("%-46s %-20s %-8s %7.3f GiB" % (k, h["shape"], h["dtype"], sz / 2**30))

# 全模型按类别归组（checkpoint 原始尺寸，非 TP 切分后）
cat = collections.Counter()
cnt = collections.Counter()
for k, sh in idx.items():
    h = head(sh)[k]
    sz = h["data_offsets"][1] - h["data_offsets"][0]
    if "visual" in k:
        c = "vision"
    elif ".experts." in k:
        c = "routed_experts"
    elif "shared_expert" in k:
        c = "shared_expert"
    elif "embed_tokens" in k:
        c = "embed"
    elif "lm_head" in k:
        c = "lm_head"
    elif "engram" in k.lower():
        c = "engram"
    else:
        c = "other(attn/dense/norm/etc)"
    cat[c] += sz
    cnt[c] += 1

print("\n=== checkpoint 全量分类（未 TP 切分）===")
tot = sum(cat.values())
for c, v in cat.most_common():
    print("%-28s %5d tensors  %8.2f GiB  %5.1f%%" % (c, cnt[c], v / 2**30, 100 * v / tot))
print("%-28s %5d tensors  %8.2f GiB" % ("TOTAL", sum(cnt.values()), tot / 2**30))