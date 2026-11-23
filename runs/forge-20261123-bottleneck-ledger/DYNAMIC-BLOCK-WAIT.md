# 动态块层证据：io_submit主要off-CPU于request tag等待

2026-11-23 goalround10。21probes20s，profile49Hz仅submitactive，sched_switch按reader tid记录offcpu；SCHEDSYNC1N8500cached0prefill4.922305642hash528b...。无重启/改参数。

8bulk层submit1591.065562ms，offcpu959.901904ms=60.3308%；prev_state0的runnable-offcpu1.170442ms=.07356%，其余onCPU残差631.163658ms。10层分记录offcpu<=submit、错误0。不要把onCPU残差算全有效工作或全浪费。

SleepStacks count11978落在schedule→io_schedule→blk_mq_get_tag→__blk_mq_alloc_requests→blk_mq_submit_bio；7个其他栈带preemption/IRQ条件，prev_state!=0不是完美睡眠类别。栈count不是时间百分比。运行栈少量样本包含tag bitmap、pin_user_pages_fast/gup、iomap/ext4、NVMe DMA/IOMMU map等；49Hz样本数有限，不按counts精确分配CPU成本。

【工程实测】submit期间大量offcpu等待，栈指认blk-mq request tag获取资源。纯CPU算力瓶颈解释不成立；不能把84.9%submit当pure softwareCPUtax。该等待是块层资源背压的证据，不证明SSD介质带宽达到峰值，也不能区分hardware tag depth、软件queue配置、突发提交或完成延迟。

下一：只读实际NVMe队列depth/nr_requests/scheduler/驱动参数、盘与Engram路径映射，匹配一次批次2048reads/4096ring、requests全接受；测完成率/实际bytes并设计submissionburst单变量对照。不能直接把nr_requests改大当既定改善，queue增大可提高延迟/并发竞争。

probeobserver风险：sched12k事件、栈捕获会放大成本。serverT接近前轮但请求不同不能证明无开销；下一必须固定corpus同长度配对控制，不把.60当无偏总体占比。GPU时间映射/worker未齐，仍不报可回收GPUidle。目标active。

原始engram-sched-trace.txt、engram-sched-summary.json、trace_engram_sched.py（已nr positive cast消warning）及SCHEDSYNC1json。CLEANUP_ABSENT True，临时container--rm，内部timeout30，单command55s。数据持久保留，无优化上线。
