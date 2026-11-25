#!/usr/bin/env python3
"""E-RECLEAN-01：造一份带 nsys 包装的 start-trace.sh（不改生产 start.sh）。

为什么必须重采：1123 trace 有两重口径污染
  ① ENGRAM_AIO=1（11-24 已改 0，空隙 17.55%→0.04%）
  ② prompt 实际 8628 token，而当前权威口径 10762 token（差 24.7%，chunk 数不同）
⇒ E-PLAN-01 的份额账不能换算，必须在当前配置+当前负载下重采。

改法沿用 1123 已验证的两处（其余配置一律不动，仍读 scripts/local.sh）：
  :337 here=(docker run ... -v "$HEAD_HF:/hf:ro")  → 追加 trace 目录挂载
  :347 here+=("${NCCL_ARGS[@]}" "$IMAGE" "${SERVE[@]}") → 换成 bash /forge-trace/wrap.sh "${SERVE[@]}"
只包 rank0（主报口径）；rank1 原样，与 1123 同法。

用法：python3 make-start-trace.py <项目根>
"""
import pathlib, sys

root = pathlib.Path(sys.argv[1])
src = root / "start.sh"
dst = root / "start-trace.sh"
t = src.read_text()

A_OLD = 'here=(docker run -d --name "$CONTAINER_NAME" $RUN_ARGS -v "$HEAD_HF:/hf:ro")'
A_NEW = 'here=(docker run -d --name "$CONTAINER_NAME" $RUN_ARGS -v "$HEAD_HF:/hf:ro" -v /home/ztxjzh/ptxwork/reclean:/forge-trace)'
B_OLD = 'here+=("${NCCL_ARGS[@]}" "$IMAGE" "${SERVE[@]}")'
B_NEW = 'here+=("${NCCL_ARGS[@]}" "$IMAGE" bash /forge-trace/wrap.sh "${SERVE[@]}")'

for old in (A_OLD, B_OLD):
    assert t.count(old) == 1, f"锚点不唯一或缺失: {old!r} 出现 {t.count(old)} 次"

t = t.replace(A_OLD, A_NEW).replace(B_OLD, B_NEW)
dst.write_text(t)
dst.chmod(0o755)

# wrap.sh：exec 让 nsys 成为 PID 1，docker stop 的 SIGTERM 才能触发它写出报告
wrap = root / "wrap-trace.sh"
wrap.write_text(
    "#!/usr/bin/env bash\n"
    "# E-RECLEAN-01 rank0 采集包装。只采 cuda-sw（GPU metrics 已由零注入 PM 单独测得，\n"
    "# 同设备不能并行两个采样器）。--sample=none --cpuctxsw=none 与 1123 同法，降低开销。\n"
    "exec /usr/local/cuda/bin/nsys profile \\\n"
    "  --trace=cuda-sw,nvtx --sample=none --cpuctxsw=none \\\n"
    "  --cuda-graph-trace=graph \\\n"
    "  --output=/forge-trace/trace-current --force-overwrite=true \\\n"
    '  "$@"\n'
)
wrap.chmod(0o755)
print("MAKE_OK start-trace.sh + wrap-trace.sh")