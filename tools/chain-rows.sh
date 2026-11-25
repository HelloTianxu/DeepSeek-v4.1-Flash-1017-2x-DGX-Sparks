#!/usr/bin/env bash
# E-ROWS-01 采集链：等 A/B 结束 → 编译期预筛 → ncu 带宽采集。
# 在容器内后台跑，日志 ~/ptxwork/rows-chain.log。
set -uo pipefail
cd /root/ptxwork

echo "=== 等待 A/B 判决 (最多 20 分钟) ==="
for i in $(seq 1 120); do
  [ -f rows-ab.json ] && break
  sleep 10
done
if [ ! -f rows-ab.json ]; then echo "TIMEOUT 无 rows-ab.json"; exit 1; fi
echo "--- rows-ab.json ---"
cat rows-ab.json

echo ""
echo "=== 编译期预筛（按 Function 块切分，避免 grep -A 串到下一函数）==="
python3 - <<'PY'
import subprocess, re, json, pathlib
out = {}
for tag, bd in [("r64", "probe-build-rows-r64"), ("r32", "probe-build-rows-r32")]:
    so = pathlib.Path(bd) / f"tf_exl3_{bd.replace('-', '_')}.so"
    if not so.exists():
        cand = list(pathlib.Path(bd).glob("*.so"))
        if not cand:
            print(f"{tag}: 无 .so"); continue
        so = cand[0]
    try:
        txt = subprocess.run(["cuobjdump", "-res-usage", str(so)],
                             capture_output=True, text=True, timeout=300).stdout
    except Exception as e:
        print(f"{tag}: cuobjdump 失败 {e}"); continue
    # 按 Function 块切分
    blocks = re.split(r"\n\s*(?=Function )", txt)
    hits = []
    for b in blocks:
        if "grouped_mma_kernel" not in b:
            continue
        name = b.split("\n")[0].strip()
        reg = re.search(r"REG:(\d+)", b)
        st = re.search(r"STACK:(\d+)", b)
        sh = re.search(r"SHARED:(\d+)", b)
        sp_s = re.search(r"SPILL_STORE:(\d+)", b)
        sp_l = re.search(r"SPILL_LOAD:(\d+)", b)
        if not reg:
            continue
        hits.append({"fn": name[:110], "reg": int(reg.group(1)),
                     "stack": int(st.group(1)) if st else 0,
                     "shared": int(sh.group(1)) if sh else 0,
                     "spill_store": int(sp_s.group(1)) if sp_s else 0,
                     "spill_load": int(sp_l.group(1)) if sp_l else 0})
    # 只看生产同款 K2=6 / CB=2 / FOLD=true / KB=4 的那一族：取 shared 最大者代表
    if hits:
        best = max(hits, key=lambda h: h["shared"])
        sh = best["shared"]
        blk_sh = 101376 // sh if sh else 99
        blk_reg = 65536 // (best["reg"] * 256) if best["reg"] else 99
        best["blocks_by_shared"] = blk_sh
        best["blocks_by_reg"] = blk_reg
        best["blocks"] = min(blk_sh, blk_reg)
        out[tag] = best
        print(f"{tag}: REG={best['reg']} spill={best['spill_store']}B shared={sh} "
              f"blocks={best['blocks']} (sh限{blk_sh} reg限{blk_reg})  [共 {len(hits)} 个变体]")
    else:
        print(f"{tag}: 未匹配到 grouped_mma_kernel")
json.dump(out, open("rows-resusage.json", "w"), indent=1)
PY

echo ""
echo "=== ncu 带宽采集（metric 集与 E-OCC-02/03 一致）==="
bash ncu-rows.sh r64 r32
echo "CHAIN_DONE"