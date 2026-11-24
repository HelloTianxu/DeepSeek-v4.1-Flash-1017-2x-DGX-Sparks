#!/usr/bin/env python3
"""Check that a self-built experts_cb2 cubin satisfies every kernel the served loader selects.

Loader contract, read from zig/src/families/dsv41/exl3_experts.zig:98-116:
  grouped_mma_kernel <CB, LO, HI, FOLD, KB>   keep CB==codebook, FOLD==1, KB==4
  grouped_rows_kernel<CB, NT, W, PF, LO, HI, G, FOLD>
                                             keep CB==codebook, NT==8, W==4, PF==1, G==2, FOLD==0
  grouped_cp_kernel  <CB, NT, W, S, LO, HI, EPI>
                                             keep CB==codebook, NT==8, W==4, S==decode_stages, EPI in 1..2
Each kept kernel fills k.mma/k.rows/k.cp over range(LO, HI); our served k2 is 6..10.
Usage: loader-coverage.py res-kit.txt res-cb2.txt [codebook] [decode_stages]
"""
import re
import sys

ARG = re.compile(r"(?:Li(-?\d+)E|Lb([01])E)")


def names(path):
    return {l.split(" ", 1)[0]: l.split(" ", 1)[1] for l in open(path).read().strip().split("\n")}


def targs(n, tag):
    i = n.find(tag + "I")
    if i < 0:
        return None
    s = n[i + len(tag) + 1:]
    out, pos = [], 0
    while True:
        m = ARG.match(s, pos)
        if not m:
            break
        out.append(int(m.group(1) if m.group(1) is not None else m.group(2)))
        pos = m.end()
    return out


def covers(t, lo_i, hi_i, need):
    return t[lo_i] <= need[0] and t[hi_i] >= need[1]


def main():
    kit, new = names(sys.argv[1]), names(sys.argv[2])
    cb = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    stages = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    need = (6, 10)  # our served k2 range
    specs = [
        ("18grouped_mma_kernel", 5, lambda t: t[0] == cb and t[3] == 1 and t[4] == 4, 1, 2, "mma"),
        ("19grouped_rows_kernel", 8,
         lambda t: t[0] == cb and t[1] == 8 and t[2] == 4 and t[3] == 1 and t[6] == 2 and t[7] == 0, 4, 5, "rows"),
        ("17grouped_cp_kernel", 7,
         lambda t: t[0] == cb and t[1] == 8 and t[2] == 4 and t[3] == stages and 1 <= t[6] <= 2, 4, 5, "cp"),
    ]
    ok = True
    for tag, nargs, pred, li, hi, label in specs:
        a = {n: t for n in kit if (t := targs(n, tag)) and len(t) == nargs and pred(t)}
        b = {n: t for n in new if (t := targs(n, tag)) and len(t) == nargs and pred(t)}
        print(f"\n{label}: kit={len(a)} mine={len(b)}  (需覆盖 k2 {need[0]}..{need[1]})")
        for n in sorted(a, key=lambda x: a[x]):
            t = a[n]
            mark = "present" if n in b else "MISSING"
            res = new.get(n, "-")
            print(f"  LO={t[li]:2} HI={t[hi]:2} covers={covers(t, li, hi, need)!s:5} {mark:8} {res}")
        for n in sorted(set(b) - set(a)):
            print(f"  only-mine {b[n]}")
        # every k2 in need must be covered by at least one present kernel
        for k2 in range(need[0], need[1] + 1):
            hit = [n for n, t in b.items() if t[li] <= k2 <= t[hi]]
            if not hit:
                print(f"  FAIL: k2={k2} uncovered in my build")
                ok = False
    print("\nCOVERAGE_OK" if ok else "\nCOVERAGE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())