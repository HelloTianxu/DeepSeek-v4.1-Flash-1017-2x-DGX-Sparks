# 稳态形状对照：专家成本不随首次库加载消失

2026-11-23用户批准10–15min2重启，head onlycuda-swtrace，worker无trace；原镜像、模型、时钟、缓存、安全守卫不变。完成后原入口恢复，两rankrunning无nsys、新CUDAerrors；SRESTORE代码512tokenhash1428664efda9一致。

三条samecorpus/rep8不同arm nonce于位置0的prompt，实际tokens8631/8628/8630，cached0max1，强首tokenhash相同。保持相同2048chunk主体形状，tailtoken数差3，非逐位相同路由。

|请求|serverprefill s|kernelwindow s|gateup ms|down ms|NCCL ms|libraryload ms/calls|
|---|---:|---:|---:|---:|---:|---:|
|1 postrestart|5.669|5.660|1287.181|626.610|550.251|926.494/25|
|2 warmshape coldprefix|4.977|4.973|1282.394|626.152|538.977|0/0|
|3 warmshape coldprefix|4.963|4.958|1290.182|633.782|507.518|0/0|

后两请求servercoldprefill约1734/1739tps，非3500。库加载0但gateup/down稳定，故首次lazyload不是稳定专家成本解释。总降约0.69s与libraryload消失相符，尾形状路由/温度不同且无多进程重复，不能把全部差值精确归因libraryload。

HtoDhostAPI累计：2827/2662/2648ms；warm后仍高。此为overlappingCUDAasynchostcall时间不可和kernel加。上一轮DMA量107MBGPU2.45ms表明相关主机阻塞非纯链路搬运，候选前序GPU同步/pageablehostbuffer/staging。

证据timeline.json steady-summary.json，selectedranges[24:6104],[6104:12184],[12184:18264]各6080kernel，按请求间idlegap且window与serverduration相符切片。head-only无workerimbalance证明。内核SASS前轮实际cubinhashb9a6...gateup/down HMMA.16816.F32非NVFP4，gateup106reg/thread256threads44KBshared/down255reg128threads32KBshared。静态资源指向decode/occupancy/memory候选，不等于stall根因。

下一步：独立微型ncu工具gate确认权限，不能productionmultipassprofiling/重放。若可用再构造保留真实路由和权重小块的隔离replay，记录DRAM/L2吞吐、Tensor/ALU管线、longscoreboard、occupancy。若计数器拒绝不改系统权限，回fallbacksingle-variablekernelmicrobench。每个新模型实验先容量账与质量/回滚。

remote ~/.cache/forge-steady-20261123/trace.nsys-rep/sqlite作为原始证据保留，exports--rm，临时wrapper及unitdropin已删除。无优化落地、3500未达。
