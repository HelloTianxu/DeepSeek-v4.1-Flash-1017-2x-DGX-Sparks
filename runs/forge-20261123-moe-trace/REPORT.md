# MoE细粒度追踪：负结果与回退验证

2026-11-23，用户批准15–20min最多2次两机重启。目标仅拆MoE，不改变生成算法。

## 实际执行

原start.sh与local.sh原文归档。创建临时启动副本start-forge-trace.sh（仅增加Nsight包装与专用trace输出卷）及临时user systemd dropin90-forge-trace.conf，原文件未修改。原镜像8348338a8d90、模型、时钟、spill、内存守卫保持。临时unit Restart=no避免追踪故障重复启动。

容器nsys2026.3.1.117；CPU sampling/ctxswitch关闭，仅CUDA/NVTX/osrt，trace90秒、kill=none。实验服务加载约35秒，但手动准备/检查消耗了采集窗口，冷prefill请求在窗口末尾到达。

## 失败1：容器PID1生命周期

Nsight90秒到期完成导出并退出，即便kill=none未主动杀目标，作为容器PID1退出仍使实验容器生命周期结束。探针未得到完整completion/usage，随后metrics连接拒绝；没有有效NTRACE原始JSON，不能声称速度或输出通过。

修正：未来wrapper必须由独立shell/监督器保持PID1，或将trace结束与应用结束明确绑定。不可只依赖kill=none。下一次必须在健康就绪时立即发请求，不允许窗口被人工审查耗尽。不得用无限duration无人值守掩盖问题。

## 失败2：没有CUDA活动数据

head保留trace.nsys-rep(3069585bytes)，离线导出sqlite成功。数据库含OSRT/NVTX，但没有CUPTI kernel/runtime/memcpy活动表。trace-table-counts与trace-diagnostics.json已归档。

diagnostic：CUDA injection initialized successfully，但No CUDA events collected、Hardware tracing used for CUDA tracing、Number CUPTI events produced10；Unified Memory trace not supported。这不足以判定具体原因。当前kernel580.178.04、容器forward compat用户driver610.43.02与Nsight硬件trace路径组合是候选，不是根因。也不能把NVTX/OSRT时间冒充kernel耗时。

因此此次没有拆清MoE486ms内部占比，MoE仍仅为上一轮干净profile的最大候选桶。没有任何新优化上线、没有性能收益声明。

## 回退已完成

立即删除经绝对路径与内容检查的90-forge-trace.conf，daemon-reload，原userunit启动；原start.sh/local.sh SHA与原文一致；两rank原命令恢复(tensorfold-native/tf-dsv41-lanes)，不含nsys。临时start-forge-trace.sh经SHA检查后删除，空dropin目录删除，不影响其他配置。

恢复后code512tokens token_sha1428664efda9、工具47tokens token_sha310581e3ca24与基线一致；约82/107serverdecode tok/s。两台运行且health正常。见head-cleanup.json与邻目录NRESTORE-code/tool-1.json。

trace专用目录作为研究证据保留(远端~/.cache/forge-moe-trace-20261123)，不是生产模型/cache目录；sqlite/read-only分析工具容器--rm已退出，无附加常驻服务。无需删除有效原始证据。

## 下一步需要先过工具门禁

1. 不再直接给生产套同一个Nsight方案。先在极小、独立、内存预算明确的CUDA程序验证本机工具能产生kernel活动表，检查hardware/software trace选项、静态/动态DriverAPI注入、当前版本compatibility。
2. 只有这一步成功，才再安排两机窗口；用保持PID1的临时包装并自动ready→request→capture停止→恢复，所有路径teardown。
3. 若CUPTI仍不支持，使用独立受控lane/进程内GPUevents做prefill专用MoE子阶段计时（routing、rotate、gateup、internal exchange、down、combine），不把eagertimer挂decodegraphs。需核实live源版本且不得直接覆盖原image。
4. 完整输出/usage/cached0和trace各自检查后，才归因。单侧累计kernel时间还要区分重叠与等待；两机clock需对齐不凭wallclock直接相减。

本轮诚实结果：工具/生命周期方案证伪，生产回退成功；微观MoE瓶颈尚未定位。先修工具门禁，再继续实验。
