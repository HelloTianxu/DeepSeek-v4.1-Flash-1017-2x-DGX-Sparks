# 阶段诊断判决：Engram供给已收敛，整体屋顶未识别

2026-11-23。本报告是基线诊断结论，不是性能优化交付。生产保持原镜像，未修改模型/量化/路由/队列，未重启。使用短时有界探针代替原拟受控重启，原研究证据保留。

## 结论先行

1. 当前prefill是复合制约：专家执行核心、其他GPU工作与Engram供给暴露等待并存，不能判为单一“硬件算力到顶”。
2. Engram供给机制有实证：小行成对直接I/O、串行双层预读、整线程join；AIO收集主导准备时间，io_submit期间大量等待blk-mq request tag。
3. 提交突发2048requests与单hwqueue1023tags匹配出可检验的软件/队列调度机制。但未测实际hwqueue分布及completion，因此不能判驱动参数错误或SSD到顶。
4. 不能保证改batch/LEAD有收益，不能把主机累计等待或跨请求比例乘到4.97s作为可回收时间。专家动态访存/算术屋顶仍未识别。

## 可审计证据链

|命题|证据类别|证据|结论边界|
|---|---|---|---|
|稳态head核心1.909s，idle0.671s|工程实测|[时间账](REPORT.md)、ledger.json|一个已有profile请求，非正式验收|
|约0.664s idle在Engram入口|工程实测+源/时序映射|gap-neighbors.json|未同请求取得ready与GPU时钟|
|串行两层reader与wholejoin|live静态核对+动态函数事件|[生产机制](LIVE-MECHANISM.md)、[动态主机](DYNAMIC-HOST.md)|join中存在后层耦合，不等于等量GPU浪费|
|AIO占准备99.17%|工程实测|[阶段账](DYNAMIC-PHASES.md)|单请求PHASESYNC1，非SSD占比|
|start占gather96.39%|工程实测|[AIO账](DYNAMIC-AIO.md)|单请求AIOSYNC1，start含回压|
|io_submit占gather84.88%|工程实测|[syscall账](DYNAMIC-SUBMIT.md)|另一请求SUBMITSYNC1，不与其他%连乘|
|submit60.33% offcpu，栈blk_mq_get_tag|工程实测|[块层账](DYNAMIC-BLOCK-WAIT.md)|SCHEDSYNC1；栈次数不是耗时比例，observer未配对|
|每层49152请求全部接受|工程实测|engram-submit-summary.json|接受不是完成；读取bytes待测|
|1023tags/queue，16queues|工程实测sysfs|[队列账](NVME-QUEUE-ACCOUNT.md)|聚合capacity非单线程capacity|
|batch2048reads约2倍单queuecapacity|理论推导|nvme-queue-account.json|completion可回收/CPU迁移未测|
|均匀residue请求32×有效bytes|理论推导+实际offset|[块几何](ENGRAM-GEOMETRY.md)|不是实际drive流量，duplicate未测|

这些比例来自不同请求与探针层级，不能组成一条无误差连乘归因链。所有请求cold prefix cached0，但nonce/文本片段不同，没有固定输入observer AB；只是机制诊断，不是无偏收益测量。

## 已排除或降级

- 不能用sparseFP4 PFLOPS衡量真实EXL3 FP16 HMMA路径。
- 全N×40层FLOP错误；实际bounded replay20N+20×128。
- HtoD host2.66s几乎全与GPU执行重叠，不是2.66s可回收复制成本。
- Engram hash/decode各毫秒级，当前不应优先大改它们。
- “最后wait很短→设备无等待”错误，submit内tagwait已测。
- 默认128浅队列不成立，当前tag1023。
- 单修对齐uniform条件只减少约3.03%请求bytes；32×不能当32×加速。
- 只核心归零且其余固定仅2816tps，3500不能只靠专家。

## 下一因果实验与停止线

先做同真实IDs/输出的submission-burst对照，而不是改整个模型：ring4096固定，batch128/256/512/1024rows；记录totalgather、accepted/completed、tagwait、actualqueue、bytes、CPU与输出byte equality。baseline/variant交错配对，先估observer噪声。

- total下降且tagwait下降：支持匹配队列的软件机制改善。
- wait换位置但total不降：收益为零，停止该方向。
- 只有多reader有效：再分离队列分散、并发、cache，不能一次混改。
- 完成瓶颈主导：减少真实重复请求/改善访问布局；需先测duplicate，不能从row大小推收益。

此对照是新的变量实验，不夹带进“只测基线”授权。每次<60s可先准备，构建/长运行另报时长；系统queue/driver/scheduler不修改。原始模型路径与临时工具隔离，退出全清理。

其次：同步GPU需求/hostready及worker时标；固定真实expertbulk的DRAM/L2/ALU/Tensor计数。无这两项不能宣布整体硬件roofline或3500可达条件最终完成。

## 当前交付状态

时间/有效算术账、live机制、供给path/tagwait、数学匹配和最小证伪实验已交付。没有新优化或可审计提速；整体目标保持未完成。最后必须核对生产健康及探针清理，再等待下一个明确因果实验。
