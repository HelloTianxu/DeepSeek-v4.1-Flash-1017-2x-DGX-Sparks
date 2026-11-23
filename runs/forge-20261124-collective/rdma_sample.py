#!/usr/bin/env python3
"""高频采样 RDMA port 计数器，测实际线上字节。

纯 sysfs 只读：不注入引擎、不碰 GPU、不分配大内存。
计数器单位：port_*_data 以 4 octet 为单位（本项目已用
data/packets×4 ≈ 1068 B/包 与 active_mtu 1024 交叉验证）。

用法：rdma_sample.py <out.csv> <duration_s> <interval_ms> [dev ...]
默认 dev：rocep1s0f1 roceP2p1s0f1
"""
import sys, time, os, pathlib

MIN_AVAIL_GIB = 4.0          # 低于此值直接拒绝启动（项目硬约束）


def avail_gib():
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576
    return 0.0


def main():
    out, dur, iv = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]) / 1000.0
    devs = sys.argv[4:] or ["rocep1s0f1", "roceP2p1s0f1"]

    a = avail_gib()
    if a < MIN_AVAIL_GIB:
        sys.exit(f"ABORT: MemAvailable {a:.2f} GiB < {MIN_AVAIL_GIB} GiB")

    paths = {}
    for d in devs:
        base = pathlib.Path(f"/sys/class/infiniband/{d}/ports/1/counters")
        if not base.is_dir():
            continue
        paths[d] = {k: base / k for k in
                    ("port_xmit_data", "port_rcv_data",
                     "port_xmit_packets", "port_rcv_packets")}
    if not paths:
        sys.exit("ABORT: no RDMA counters found")

    t0 = time.monotonic()
    n = 0
    with open(out, "w") as f:
        f.write("t,wall,dev,xmit_data,rcv_data,xmit_pkt,rcv_pkt\n")
        while True:
            t = time.monotonic() - t0
            if t > dur:
                break
            wall = time.time()
            for d, ps in paths.items():
                vals = []
                for k in ("port_xmit_data", "port_rcv_data",
                          "port_xmit_packets", "port_rcv_packets"):
                    try:
                        vals.append(int(ps[k].read_text().strip()))
                    except OSError:
                        vals.append(-1)
                f.write(f"{t:.6f},{wall:.6f},{d},{vals[0]},{vals[1]},{vals[2]},{vals[3]}\n")
            n += 1
            time.sleep(iv)
    print(f"sampled {n} ticks over {dur}s -> {out} (avail {a:.2f} GiB)")


if __name__ == "__main__":
    main()