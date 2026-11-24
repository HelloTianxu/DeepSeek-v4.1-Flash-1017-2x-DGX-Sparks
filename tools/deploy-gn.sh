#!/usr/bin/env bash
# E-SELF-04b deployment: one image + one cubin serve every arm; the A/B is an env flip.
#
# Why one cubin is enough: the superset cubin's baseline G=2 kernels were verified byte-for-byte
# against the kit (4096 SASS instructions each, both served k2 ranges), so the control arm runs the
# same bits it runs today. G=4/G=8 are added template instantiations, selected by TF_DS_EXPERT_G.
#
# The kit cubin lives only on head: the worker's dsv41-1017-exl3-hf volume is NFS from
# 192.168.100.10:/home/ztxjzh/.cache/huggingface, mounted read-only. Replacing it on head therefore
# reaches both ranks, and their cubins fingerprint agrees by construction.
# start.sh does not hash the kit (only prepare.sh does); worker_files_check only tests presence.
set -eu
IMG=tensorfold-dsv41:base-gn
KIT=$HOME/.cache/huggingface/tensorfold-dsv41/kit-fdde3285/kit/cubins
W=$HOME/ptxwork
WORKER=ztxjzh@192.168.100.11

echo "== 1. 产物内容断言 =="
for b in "$HOME/build-alias-out/native/bin/tensorfold-native" "$HOME/build-alias-out/bin/tf-dsv41-lanes"; do
  test -s "$b" || { echo "FAIL missing $b"; exit 1; }
  n=$(strings "$b" | grep -c '^TF_DS_EXPERT_G$' || true)
  [ "$n" -ge 1 ] || { echo "FAIL: $b does not contain TF_DS_EXPERT_G"; exit 1; }
  echo "  $(basename "$b") contains TF_DS_EXPERT_G"
done
test -s "$W/experts_cb_gn.cubin" || { echo "FAIL missing superset cubin"; exit 1; }

echo "== 2. head 镜像 =="
mkdir -p "$HOME/img-gn"
cp "$HOME/build-alias-out/native/bin/tensorfold-native" "$HOME/build-alias-out/bin/tf-dsv41-lanes" "$HOME/img-gn/"
cat > "$HOME/img-gn/Dockerfile" <<'EOF'
FROM tensorfold-dsv41:base-alias
COPY tensorfold-native /opt/tensorfold/native/bin/tensorfold-native
COPY tf-dsv41-lanes /opt/tensorfold/bin/tf-dsv41-lanes
LABEL tf.engine=baseline-alias-gn tf.exp=E-SELF-04b
EOF
docker build -q -t "$IMG" "$HOME/img-gn" >/dev/null
docker image inspect "$IMG" >/dev/null && echo "  head $IMG built"

echo "== 3. worker 镜像 =="
ssh -o ConnectTimeout=10 "$WORKER" 'mkdir -p ~/img-gn'
scp -q "$HOME/img-gn/Dockerfile" "$HOME/img-gn/tensorfold-native" "$HOME/img-gn/tf-dsv41-lanes" "$WORKER:~/img-gn/"
ssh "$WORKER" "docker build -q -t $IMG ~/img-gn >/dev/null && docker image inspect $IMG >/dev/null && echo '  worker $IMG built'"

echo "== 4. 安装超集 cubin（仅 head，worker 经 NFS 共享）=="
if [ ! -f "$KIT/experts_cb.cubin.kit-orig" ]; then
  cp -p "$KIT/experts_cb.cubin" "$KIT/experts_cb.cubin.kit-orig"
  echo "  原件已备份 experts_cb.cubin.kit-orig"
else
  echo "  备份已存在，不覆盖"
fi
sha256sum "$KIT/experts_cb.cubin.kit-orig" | sed 's/^/  orig /'
cp "$W/experts_cb_gn.cubin" "$KIT/experts_cb.cubin"
sha256sum "$KIT/experts_cb.cubin" | sed 's/^/  new  /'
ls -la "$KIT/experts_cb.cubin" "$KIT/experts_cb.cubin.kit-orig" | sed 's/^/  /'

echo "== 5. worker 经 NFS 看到同一文件 =="
ssh "$WORKER" "docker run --rm --network none -v dsv41-1017-exl3-hf:/hf:ro --entrypoint sha256sum $IMG /hf/tensorfold-dsv41/kit-fdde3285/kit/cubins/experts_cb.cubin" | sed 's/^/  worker /'
echo DEPLOY_GN_DONE