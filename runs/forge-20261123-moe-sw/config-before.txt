# Two-node DGX Spark TP2 settings.
WORKER=ztxjzh@192.168.100.11
FABRIC_PEER=192.168.100.11
NFS_SERVER=192.168.100.10
NCCL_RAILS=2
RDMA=1
EAQ_V1_1=1
ABLIT=1
export HF_HUB_OFFLINE=1

# LAN API and the existing conservative memory profile.
HOST=0.0.0.0
PORT=8888
CONTEXT=524288
SPILL_GIB=64
KV_SHARD=1
HEAD_ENGRAM_DIR=/home/ztxjzh/.cache/dsv41-tensorfold/engram-1bae0e10

# Require authentication on every endpoint except minimal health.
API_KEY=1

# Forge controlled prefill experiment; original config backed up beside this file.
PREFILL_INTERLEAVE_MS=0
export TF_DS_PREFILL_INTERLEAVE_MS=0


