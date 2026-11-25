#!/usr/bin/env python3
"""E-FEED-02：把 engram upload 的源缓冲改成 page-locked（TF_DS_ENGRAM_PINNED，默认关）。

为什么新增 pinRows 而不是改 init 签名：
  EngramHost.init 有 6 处调用点，其中 4 处是 engram_lead.zig 的单元测试（无 CUDA driver）。
  改签名会波及测试。新增独立方法只在 model.zig 的生产路径调用，测试路径零影响。

为什么可以直接替换 slice 而不拷旧内容：
  init 后这些缓冲未初始化；engramAhead1 / decodeRows 都是先写满再 upload，不存在先读后写。

位一致性：只换内存来源，不动任何数值 ⇒ 输出应逐 token 相同。
"""
import sys
import pathlib

ROOT = pathlib.Path(sys.argv[1])
P = ROOT / "zig/src/families/dsv41/prompt.zig"
M = ROOT / "zig/src/families/dsv41/model.zig"

FIELD_OLD = "    rows: []u16, // bf16 [cap, (hi - lo) * head_dim]\n"
FIELD_NEW = (
    "    rows: []u16, // bf16 [cap, (hi - lo) * head_dim]\n"
    "    pinned_bytes: usize = 0, // TF_DS_ENGRAM_PINNED: page-locked upload sources (E-FEED-02)\n"
)

ANCHOR = "    /// TF_DS_ENGRAM_LEAD: each Engram layer's prompt reader gets a hash buffer of its own and its ring, rings[j].\n"
METHOD = '''    /// TF_DS_ENGRAM_PINNED=1: the upload's sources become page-locked, so the async copy needs no staging.
    /// E-FEED-01: a pageable source turns cuMemcpyHtoDAsync into a throttle point -- the driver stages it in
    /// chunks and the caller's thread is paced by the GPU's consumption (24 MiB took 375-642 ms of host time
    /// while the GPU was 100% busy). Pinned memory is the only host memory an async copy reads without staging.
    /// Runs once at open; the pinned bytes live for the process, like the tables and the reader pool. The
    /// arena's originals are left alone (nothing reads them before being overwritten), so this costs one
    /// extra set of buffers, not a leak per request.
    pub fn pinRows(eh: *EngramHost, d: *const cuda.Driver) !void {
        const v = std.c.getenv("TF_DS_ENGRAM_PINNED") orelse return;
        if (std.mem.eql(u8, std.mem.span(v), "0")) return;
        var total: usize = 0;
        const rb = eh.rows.len * 2;
        eh.rows = (try cuda.HostBuffer.alloc(d, rb)).slice(u16);
        total += rb;
        for (&eh.ahead_rows) |*ar| {
            if (ar.len == 0) continue;
            const b = ar.len * 2;
            ar.* = (try cuda.HostBuffer.alloc(d, b)).slice(u16);
            total += b;
        }
        eh.pinned_bytes = total;
        std.log.info("dsv41: Engram upload buffers page-locked, {d} MiB (TF_DS_ENGRAM_PINNED)", .{total >> 20});
    }

'''

CALL_OLD = "            m.eh = try prompt.EngramHost.init(a, c, engram.Hasher.init(c.*, map, mult), &m.tables.?, m.epool.?, sp.rank, sp.world, chunk_rows);\n"
CALL_NEW = CALL_OLD + "            if (m.eh) |*eh| try eh.pinRows(d);\n"


def sub(path, old, new, label):
    t = path.read_text()
    n = t.count(old)
    assert n == 1, f"{label}: expected 1 occurrence, got {n}"
    assert new not in t, f"{label}: already applied"
    path.write_text(t.replace(old, new))
    print(f"  ok {label}")


def main():
    for f in (P, M):
        assert f.is_file(), f"missing {f}"
    sub(P, FIELD_OLD, FIELD_NEW, "prompt.zig field pinned_bytes")
    sub(P, ANCHOR, METHOD + ANCHOR, "prompt.zig pinRows method")
    sub(M, CALL_OLD, CALL_NEW, "model.zig pinRows call")
    t = P.read_text()
    assert t.count("pub fn pinRows") == 1
    assert "cuda.HostBuffer.alloc" in t
    print("PATCH_OK")


if __name__ == "__main__":
    main()