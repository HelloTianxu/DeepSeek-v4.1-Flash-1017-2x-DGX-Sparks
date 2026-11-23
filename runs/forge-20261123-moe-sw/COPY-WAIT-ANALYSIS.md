# Round3：复制物理成本与主机等待分离

2026-11-23，本轮不重启生产，不改配置。worker独立4MiB CUDA门禁成功，100kernel、207runtime完整活动记录；工具可用，真实follower报告缺失CUDA不能解释为节点不支持。证据worker-cuda-sw-gate.txt；测试--rmnetworknone容器退出，原生产容器独存，无nsys残留。

## 真实head请求内复制

以此前5.632s coldprefillkernel窗口切片CUPTI memcpy：

|方向|bytes|GPU累计ms|calls|
|---|---:|---:|---:|
| HtoD |106750464|2.449427|505|
| DtoD |6685696|0.144833|47|
| DtoH |13975552|0.371874|65|

证据head-copy-detail/summary.json。比较head-api-summary中的HtoD主机累计2804ms：主机调用阻塞与实际DMA时间差三个数量级。不能把2.8s解释成HtoD带宽差，亦不能把它全称CPU计算；等待先前GPU、pageablebuffer staging、lazyload等仍为候选。API和GPU重叠不得收益相加。

cuLibraryLoadData累计952ms，其中857.575ms与内核union的无kernel空隙重叠（head-gap-correlation.json）。时间相关不是可消除gain证明。首次postrestart形状加载可能主导这部分；需要同进程形状warm、另一noncecached0coldprompt控制。不能凭此宣称稳定多出858ms或把优化值按比例加到MoE。

## 修正下一步

1. 真trace进程内先预热相同8K/16K形状；第二条相同主体长度不同nonceprompt，cached0，记录首条/第二条所有库加载与kernel分解。
2. 只tracehead先取得稳态，worker另使用正常进程退出cuProfilerStop/flush策略审查，或prefill-onlyevents子阶段。独立workergate证明需查follower特异采集收尾，不去改驱动或系统权限。
3. 以SASS/ncu测EXL3gateup/down解码与memorytensorstall，必须先工具/容量gate，不能对服务进行multipasskernelreplay。

本轮主战场继续聚焦专家gateup/down1.88s以及首形状库加载/等待；真实稳定瓶颈仍需warmcontrol。3500尚未达，生产保持健康。
