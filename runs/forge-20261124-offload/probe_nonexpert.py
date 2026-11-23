import json, struct, collections, re

M = "/home/ztxjzh/.cache/huggingface/tensorfold-dsv41/aee640e2-ablit-0bdc2036/model"
idx = json.load(open(M + "/model.safetensors.index.json"))["weight_map"]
hdr = {}
def head(sh):
    if sh not in hdr:
        with open(M + "/" + sh, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr[sh] = json.loads(f.read(n))
    return hdr[sh]

# 非 routed_experts 的张量，按去掉层号的模式归组
pat = collections.Counter()
size = collections.Counter()
sample = {}
for k, sh in idx.items():
    if ".experts." in k:
        continue
    h = head(sh)[k]
    sz = h["data_offsets"][1] - h["data_offsets"][0]
    p = re.sub(r"\.\d+\.", ".N.", k)
    pat[p] += 1
    size[p] += sz
    if p not in sample:
        sample[p] = (h["shape"], h["dtype"])

print("=== 非 routed_experts 张量模式（共 %d 个）===" % sum(pat.values()))
for p, v in size.most_common(40):
    shp, dt = sample[p]
    print("%-58s x%-4d %-18s %-7s %7.3f GiB" % (p, pat[p], str(shp), dt, v / 2**30))
print("非专家合计 %.3f GiB" % (sum(size.values()) / 2**30))