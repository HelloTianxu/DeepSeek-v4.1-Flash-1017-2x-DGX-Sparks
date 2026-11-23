# 动态阶段归因：Engram供给时间99.17%在AIO收集路径

2026-11-23，goal round7。无重启20s有界12函数探针，READY同步单请求。原镜像/模型/路由不变，无新优化。

PHASESYNC1: N8500cached0，serverprefill4.908505666s，hash528b0f3da89f与baseline首token一致。单请求不同nonce/corpus片段，无observer配对控制，不报提速。

8个2048row层级reader总准备1899.519ms，其中aioGather1883.810ms，占99.173%。hash在j0一次计算双层，0.515–0.671ms；decode单层1.315–1.851ms。aioGather224.033–248.942ms。tail308row分别44.780/36.556ms，aio43.951/36.023ms。解析器强制10个reader、phaseentryexit全配对无残项。

【工程实测】CPU哈希/FP8decode不是这次Engram供给主要耗时。AIO收集路径是供给主要制约；它包括start提交、ring回压、kernelI/O、wait/reap、copy及分配，不等于纯SSD。

【研究假设】需要拆submit/start vs wait/reap与actualrequestspans。下一采样可用已有Aio.start/Aio.wait/Aio.reap函数边界累计inclusive/exclusive，避免把nested时间相加。所有block事件尽量离线，先少数batch区分提交慢还是完成等待慢，再量实际unique块/bytes。

GPU关键路径收益仍未闭合：前轮host双层join约470ms不等于GPUidle，可与前chunkGPU重叠。8层累计供给不能从4.908s中全扣除。两rank供给与collective就绪需对齐。

证据engram-phase-trace.txt、engram-phase-summary.json、trace_engram_phases.py、parse_engram_phases.py与PHASESYNC1请求JSON。capture返回0、CLEANUP_ABSENT True。各command<=55s，容器内部timeout30s，BPFinterval20s；原服务未重启。rootcause范围收紧为EngramAIOpath+whole-reader耦合，尚未判SSD物理上限或driver/software缺陷。
