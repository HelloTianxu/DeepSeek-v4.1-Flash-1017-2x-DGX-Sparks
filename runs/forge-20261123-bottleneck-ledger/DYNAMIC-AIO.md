# 动态AIO内部账：start96.39%，但不可判纯提交开销

2026-11-23 goal round8。无重启20s16函数探针，READY同步AIOSYNC1请求N8559cached0、serverprefill4.912537643s、hash528b0f3da89f。不改变模型/路由/系统配置。

8个2048row层级gather累计1891.625377ms：Aio.start1823.367777ms（96.39159%）、Aio.wait67.582986ms（3.57275%），其他0.674614ms。每层start24calls/wait24calls，与24576行分24批1024行吻合；尾367rows各5calls。全部10层记录配对、inclusive加法闭合、残差>=0。

单层start221.765–237.068ms，最后wait5.387–12.133ms。这表明大部分成本在提交阶段函数范围，而不是最后wait；不能说明io_submit syscall单独耗221ms。

代码start内联submitLocked：ring槽位被前lap占用时在while中reap，之后才分配/构造iocb/ioSubmit。因此start含完成回压、reap/copy、分配、kernel提交等。若真实I/O完成慢，即使最后wait短，也可能在start里等。下一需分离actualio_submit耗时及提交轮数、ringbackpressure/reap；不得把start96%当“纯框架税”。

本轮没有对Aio.reap每次调用挂probe：它在busy polling高频执行，探针可能放大observer成本。采取低频24start/24wait汇总，每层一次输出，避免每49152read或每poll打印。后续优先聚合syscall io_submit/io_getevents与完成数、start时间内部实际占比；必要时小窗抽样，不全量poll探针。

现有128/2048shape/字节账与Engramwholejoin仍有效；这份是单head单请求无GPU时间映射，不能从4.912s减累计1.823s作为收益，也不能判SSD到顶。collective/worker就绪与observer配对未测。

证据：trace_engram_aio.py、engram-aio-trace.txt、engram-aio-summary.json、AIOSYNC1-prefill8-35.json。清理返回CLEANUP_ABSENT True，命令<=55s，内部timeout30s/BPF20s；原服务未重启。原始数据永久保留。
