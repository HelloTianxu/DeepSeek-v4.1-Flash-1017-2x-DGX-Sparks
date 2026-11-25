#!/usr/bin/env python3
"""按模板参数精确匹配生产变体做编译期预筛。

教训：上一版取「shared 最大者」，抓到别的变体，报 r64 REG=133/blocks=1，
与 r64 实测 warp 33.20%（2 blocks）矛盾。生产变体由 ncu 内核名核实为
grouped_mma_kernel<2, 2, 10, 1, 4>（CB=2, LO=2, HI=10, FOLD=true, KB=4）。
预筛与 ncu 一样必须先验内核身份（手册 §7.26 的推广）。
"""
import subprocess, re, json, pathlib, sys

PAT = sys.argv[1] if len(sys.argv) > 1 else "grouped_mma_kernel<2, 2, 10, 1, 4>"
TREES = [("r64", "probe-build-rows-r64"), ("r32", "probe-build-rows-r32")]
out = {}
for tag, bd in TREES:
    p = pathlib.Path(bd)
    if not p.exists():
        print(f"{tag}: 无目录 {bd}")
        continue
    sos = list(p.glob("*.so"))
    if not sos:
        print(f"{tag}: 无 .so")
        continue
    txt = subprocess.run(["cuobjdump", "-res-usage", str(sos[0])],
                         capture_output=True, text=True).stdout
    hit = None
    for b in re.split(r"\n\s*(?=Function )", txt):
        if PAT not in b:
            continue
        def g(k):
            m = re.search(k + r":(\d+)", b)
            return int(m.group(1)) if m else 0
        reg, sh = g("REG"), g("SHARED")
        bs = 101376 // sh if sh else 99
        br = 65536 // (reg * 256) if reg else 99
        hit = {"reg": reg, "shared": sh, "spill_store": g("SPILL_STORE"),
               "spill_load": g("SPILL_LOAD"), "stack": g("STACK"),
               "blocks": min(bs, br), "by_shared": bs, "by_reg": br}
        break
    if hit:
        out[tag] = hit
        print(f"{tag}: REG={hit['reg']} spill={hit['spill_store']}B shared={hit['shared']} "
              f"blocks={hit['blocks']} (sh限{hit['by_shared']} reg限{hit['by_reg']})")
    else:
        print(f"{tag}: 未匹配 {PAT}")
json.dump(out, open("rows-resusage-fixed.json", "w"), indent=1)
print("WROTE rows-resusage-fixed.json")