#!/usr/bin/env python3
"""E-OCC-01：让 grouped_mma 的 KB 可运行时覆盖，实现「只动 KB、不动 SK/warps」的干净单因素。

背景（三次干预未生效的教训）：
  pybind 签名是 mats,K,N,P,SK,slots,cb,**warps**,lo,hi,fold —— 没有 kb 参数。
  KB 是编译期模板参数，由 dispatch 从 chain 自动推：
      chain = (K/16)/(SK*warps)；kb = chain%4==0 ? 4 : (pf&2 && chain%6==0) ? 6 : (pf&1 && chain%2==0) ? 2 : 0
  生产 gate/up：K=5120, SK=4, warps=4 ⇒ chain=20 ⇒ 20%4==0 ⇒ KB=4（shared 44288）。
  想通过参数改 KB 只能改 SK/warps，但那会同时改 K 链划分与每 warp 工作量 ⇒ 混淆因素。
本 patch 加 TF_MMA_KB 环境变量覆盖（每次 launch 读，非 static ⇒ 同进程可切换），
并断言 chain%kb==0 保证合法。源码注释保证「每条链的 k tile 顺序不随 step 变」⇒ 应位一致，可验证。

用法：python3 patch-mma-kb.py <tree_root>   # 幂等
"""
import pathlib, sys

root = pathlib.Path(sys.argv[1])
p = root / "experts_grouped.cuh"
t = p.read_text()
if "TF_MMA_KB_APPLIED" in t:
    print("PATCH_ALREADY_APPLIED"); sys.exit(0)

OLD = ('    const int kb = chain % MMA_KB == 0 ? 4 : ((a.pf & 2) && chain % 6 == 0) ? 6 : '
       '((a.pf & 1) && chain % 2 == 0) ? 2 : 0;\n'
       '    TORCH_CHECK(kb != 0, "prompt mma: chains of ", chain, " k tiles take no allowed step");')
NEW = ('    // TF_MMA_KB_APPLIED: runtime override so KB can be A/B-tested without touching SK/warps.\n'
       '    const int kb_auto = chain % MMA_KB == 0 ? 4 : ((a.pf & 2) && chain % 6 == 0) ? 6 : '
       '((a.pf & 1) && chain % 2 == 0) ? 2 : 0;\n'
       '    int kb = kb_auto;\n'
       '    { const char* v = getenv("TF_MMA_KB");\n'
       '      if (v && v[0]) { int want = atoi(v);\n'
       '          TORCH_CHECK(want == 2 || want == 4 || want == 6, "TF_MMA_KB must be 2/4/6");\n'
       '          TORCH_CHECK(chain % want == 0, "TF_MMA_KB=", want, " does not divide chain ", chain);\n'
       '          kb = want; } }\n'
       '    TORCH_CHECK(kb != 0, "prompt mma: chains of ", chain, " k tiles take no allowed step");')
assert t.count(OLD) == 1, f"锚点计数 {t.count(OLD)}"
p.write_text(t.replace(OLD, NEW))
print("PATCH_OK")