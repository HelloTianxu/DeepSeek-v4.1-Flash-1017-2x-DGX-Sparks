#!/usr/bin/env python3
"""E-DRAM-01：从 DRY_RUN 原文生成两个 rank 的启动脚本。

纪律：命令原样复制，rank 0 只做两处增量——
  ① --cap-add IPC_LOCK 后加 --cap-add SYS_ADMIN（取 profiling 权限，不改 host 驱动）
  ② CMD 的 tensorfold-native serve 前插入 ncu
  ③ 加 -v /tmp:/tmp 让 ncu 的 CSV 落到 host
rank 1 完全原样。镜像 ENTRYPOINT 是 nvidia_entrypoint.sh（banner 后 exec "$@"），
所以插入 ncu 不需要覆盖 entrypoint。

用法：python3 gen_launch.py /tmp/dryrun.txt <launch_count> <launch_skip>
输出：/tmp/rank1.cmd  /tmp/rank0.cmd
"""
import sys, re, pathlib

dry, count, skip = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
lines = pathlib.Path(dry).read_text().splitlines()

def cmd_after(marker):
    for i, l in enumerate(lines):
        if marker in l:
            # 命令在标记行的下一行（可能缩进）
            return lines[i + 1].strip()
    raise SystemExit(f"marker not found: {marker}")

r1 = cmd_after("[dry-run] rank 1 on")
r0 = cmd_after("[dry-run] rank 0 here:")

IMAGE = "tensorfold-dsv41:exl3-tp2-8348338a8d90"
assert r0.count(f"{IMAGE} tensorfold-native serve") == 1, "rank0 CMD 锚点不唯一"
assert "--cap-add IPC_LOCK" in r0 and "--cap-add IPC_LOCK" in r1

# GB10 上校准过的 DRAM 读口径（见 REPORT §2）：op_read_lookup_miss × 32B。
# op_read 是 L2 请求，两者之比即重读是否被 L2 吸收。
METRICS = ",".join([
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_read_lookup_miss.sum",
    "lts__t_sectors_op_write.sum",
    "lts__t_sectors_op_write_lookup_miss.sum",
    "sm__inst_executed.sum",
])
NCU = (
    "ncu --target-processes all --graph-profiling node --clock-control none "
    f"--kernel-name regex:grouped_mma_kernel --launch-count {count} --launch-skip {skip} "
    f"--metrics {METRICS} --csv --log-file /tmp/ncu-experts.csv"
)

r0n = r0.replace("--cap-add IPC_LOCK", "--cap-add IPC_LOCK --cap-add SYS_ADMIN", 1)
r0n = r0n.replace("--device /dev/infiniband", "--device /dev/infiniband -v /tmp:/tmp", 1)
r0n = r0n.replace(f"{IMAGE} tensorfold-native serve", f"{IMAGE} {NCU} tensorfold-native serve", 1)

# 自检：增量必须恰好三处，其余逐字不变
back = r0n.replace(" --cap-add SYS_ADMIN", "", 1).replace(" -v /tmp:/tmp", "", 1).replace(f"{NCU} ", "", 1)
assert back == r0, "rank0 除三处增量外被改动了"

pathlib.Path("/tmp/rank1.cmd").write_text(r1 + "\n")
pathlib.Path("/tmp/rank0.cmd").write_text(r0n + "\n")
print("rank1 bytes", len(r1), "rank0 bytes", len(r0n))
print("NCU:", NCU)