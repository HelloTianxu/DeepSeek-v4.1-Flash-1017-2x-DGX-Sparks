#!/usr/bin/env bash
# E-SELF-04 bit-identity gate.
# Extract the PTX embedded in the kit cubin (.nv_debug_ptx_txt), re-assemble with the
# same toolkit, then compare SASS + resource usage for EVERY tf_exl3x kernel.
#
# Whole-file byte identity is NOT the gate: ptxas cannot reproduce the kit's .debug_*
# sections (kit 49.5 MB vs rebuilt ~11.7 MB). The gate is instruction identity + resource
# identity, which is what determines behaviour and performance.
#
# Every comparison asserts non-empty inputs first. An empty extract is a FAIL, never an
# "IDENTICAL" (same failure class as the grep -c BUILD_OK false positive, handbook 7.5).
set -eu
K=/hf/tensorfold-dsv41/kit-fdde3285/kit/cubins
W=/work
SRC=$K/experts_cb.cubin

echo "== 1. 抽取内嵌 PTX =="
python3 - "$SRC" "$W/kit.ptx" <<'PY'
import struct, sys
p, out = sys.argv[1], sys.argv[2]
b = open(p, 'rb').read()
e_shoff = struct.unpack_from('<Q', b, 0x28)[0]
e_shentsize = struct.unpack_from('<H', b, 0x3a)[0]
e_shnum = struct.unpack_from('<H', b, 0x3c)[0]
e_shstrndx = struct.unpack_from('<H', b, 0x3e)[0]
sh = lambda i: struct.unpack_from('<IIQQQQIIQQ', b, e_shoff + i * e_shentsize)
so = sh(e_shstrndx)[4]
nm = lambda o: b[so + o:b.index(b'\0', so + o)].decode()
for i in range(e_shnum):
    s = sh(i)
    if nm(s[0]) == '.nv_debug_ptx_txt':
        data = b[s[4]:s[4] + s[5]]
        open(out, 'wb').write(data.replace(b'\x00', b'\n'))
        print('  ptx bytes', len(data))
        break
else:
    sys.exit('FAIL: no .nv_debug_ptx_txt section')
PY

echo "== 2. 往返汇编 =="
time ptxas -arch=sm_121 -o "$W/rebuilt.cubin" "$W/kit.ptx" 2> "$W/ptxas.log" || {
  echo "FAIL_PTXAS"; tail -20 "$W/ptxas.log"; exit 1; }
test -s "$W/rebuilt.cubin" || { echo FAIL_EMPTY_CUBIN; exit 1; }
echo "  rebuilt $(stat -c%s "$W/rebuilt.cubin") B / kit $(stat -c%s "$SRC") B"

echo "== 3. 资源账（全部 tf_exl3x 内核）=="
for tag in kit rebuilt; do
  f=$SRC; [ "$tag" = rebuilt ] && f=$W/rebuilt.cubin
  cuobjdump -res-usage "$f" 2>/dev/null | awk '
    $1=="Function" && $2 ~ /tf_exl3x/ {name=$2; sub(/:$/,"",name); next}
    name!="" && $1 ~ /^REG:/ {print name, $1, $2, $3; name=""}' \
    | sort > "$W/res-$tag.txt"
  n=$(wc -l < "$W/res-$tag.txt")
  echo "  $tag: $n kernels"
  test "$n" -gt 50 || { echo "FAIL_RESUSAGE_EXTRACT ($n)"; exit 1; }
done
if cmp -s "$W/res-kit.txt" "$W/res-rebuilt.txt"; then
  echo "  VERDICT RESUSAGE_IDENTICAL ($(wc -l < "$W/res-kit.txt") kernels)"
else
  echo "  VERDICT RESUSAGE_DIFFERS"
  diff "$W/res-kit.txt" "$W/res-rebuilt.txt" | head -30
fi

echo "== 4. 全内核 SASS 逐条对照 =="
for tag in kit rebuilt; do
  f=$SRC; [ "$tag" = rebuilt ] && f=$W/rebuilt.cubin
  # keep only instruction text; drop /*addr*/ and the trailing encoding words
  cuobjdump -sass "$f" 2>/dev/null \
    | sed -n 's@^[[:space:]]*/\*[0-9a-f]\{4\}\*/[[:space:]]\+\(.*\);.*$@\1@p' \
    | sed 's/[[:space:]]\+/ /g; s/[[:space:]]*$//' > "$W/sass-all-$tag.txt"
  n=$(wc -l < "$W/sass-all-$tag.txt")
  echo "  $tag: $n instrs"
  test "$n" -gt 10000 || { echo "FAIL_SASS_EXTRACT ($n)"; exit 1; }
done
if cmp -s "$W/sass-all-kit.txt" "$W/sass-all-rebuilt.txt"; then
  echo "  VERDICT SASS_IDENTICAL ($(wc -l < "$W/sass-all-kit.txt") instrs, all kernels)"
else
  echo "  VERDICT SASS_DIFFERS: $(diff "$W/sass-all-kit.txt" "$W/sass-all-rebuilt.txt" | grep -c '^[<>]') lines"
  diff "$W/sass-all-kit.txt" "$W/sass-all-rebuilt.txt" | head -20
fi

echo "== 5. 我方 served 两件单独判定 =="
for V in 16 10; do
  M=$(awk -v v="ILi2ELi8ELi4ELi1ELi2ELi${V}ELi2ELb0" \
      '$1 ~ /grouped_rows_kernel/ && index($1,v) {sub(/^REG:/,"",$1); print $1}' "$W/res-kit.txt" | head -1)
  NAME=$(awk -v v="ILi2ELi8ELi4ELi1ELi2ELi${V}ELi2ELb0" \
      'index($1,"grouped_rows_kernel") && index($1,v) {print $1}' "$W/res-kit.txt" | head -1)
  test -n "$NAME" || { echo "  HI=$V FAIL_NAME_NOT_FOUND"; continue; }
  echo "  HI=$V kit:     $(grep -F "$NAME" "$W/res-kit.txt" | cut -d' ' -f2-)"
  echo "  HI=$V rebuilt: $(grep -F "$NAME" "$W/res-rebuilt.txt" | cut -d' ' -f2-)"
  for tag in kit rebuilt; do
    f=$SRC; [ "$tag" = rebuilt ] && f=$W/rebuilt.cubin
    cuobjdump -sass -fun "$NAME" "$f" 2>/dev/null \
      | sed -n 's@^[[:space:]]*/\*[0-9a-f]\{4\}\*/[[:space:]]\+\(.*\);.*$@\1@p' \
      | sed 's/[[:space:]]\+/ /g; s/[[:space:]]*$//' > "$W/s-$V-$tag.txt"
    test -s "$W/s-$V-$tag.txt" || { echo "  HI=$V FAIL_SASS_EMPTY ($tag)"; exit 1; }
  done
  if cmp -s "$W/s-$V-kit.txt" "$W/s-$V-rebuilt.txt"; then
    echo "  HI=$V VERDICT IDENTICAL ($(wc -l < "$W/s-$V-kit.txt") instrs)"
  else
    echo "  HI=$V VERDICT DIFFERS ($(diff "$W/s-$V-kit.txt" "$W/s-$V-rebuilt.txt" | grep -c '^[<>]') lines)"
    diff "$W/s-$V-kit.txt" "$W/s-$V-rebuilt.txt" | head -10
  fi
done

echo "== 6. 整文件字节（仅记录，非判据）=="
sha256sum "$SRC" "$W/rebuilt.cubin" | sed 's/^/  /'
echo GATE_DONE