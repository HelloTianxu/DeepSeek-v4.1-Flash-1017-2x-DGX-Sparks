# MoE时间线实验预注册

目标：拆清MoE桶内部的EXL3group/rot_in/gateup/down/combine、NCCL交换、CUDA API等待。不是速度优化，不用instrumentedTPS宣称达标。

窗口用户批准15–20min最多2次两机重启。原脚本未改，原config未改，原镜像8348338a8d90、模型、cache、spill、guards保持。临时start-forge-trace.sh为原start.sh副本，只加nsys与trace输出卷；userunit临时90-forge-trace.conf仅改ExecStart到副本及Restart=no。恢复必须删除该特定dropin、daemon-reload、原unit启动、核对启动原文/configSHA及生成hash。删除前核实绝对目标，不能碰其他dropins。

Nsight容器版本2026.3.1.117，trace CUDA/NVTX/osrt，CPU sampling/ctxswitch关闭；duration90秒，kill=none（默认sigterm不能用），应用正常停止后flush。启动加载也被记录，必须按实际请求时间窗口切片排除。只送一个冷prefill max1，无decode。原生CPU sampling被perf paranoid4拒绝，本实验不改变系统权限。

原始nsys-rep与sqlite要保存远端专门cache目录，导出分析仅复制所需小统计。不要下载SDK/build/权重，不自动扩大维护。结束后验证两rank运行，无Nsight/profiling覆盖，health与生成一致；临时副本/系统覆盖去除，trace retained为研究交付。
