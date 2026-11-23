# 队列账与下一单变量实验

2026-11-23 goalround11。只读NVMe/sysfs、Engrammount配置，不调整系统或模型。

## 真实配置

Engram位于/dev/nvme0n1p2，ext4 rw,relatime。SSD型号YMTC PC450-2TB-D。scheduler none（可选mq-deadline），nomerges0；nr_requests1023。NVMe io_queue_depth1024，poll_queues0、write_queues0、threaded_interrupts0。16hardwarequeues，每queue nr_tags1023/reserved0，CPU映射见原始JSON。

因此“默认128浅队列”不是本机原因，直接把nr_requests增到1024不解决问题；当前本来已1023。读ahead512KB不是O_DIRECT随机reads的有效数据缓存证明。

## 数学匹配

当前每batch1024Engramrows=>2048read requests，24batch=>49152requests。单queue1023tags：burst/capacity2.00196；若无完成复用且全部映射单queue，至少1025请求不能同时占tag。ring4096reads/capacity4.00391。每layer请求相当48.0469个单queuecapacity周转。

这与dynamicio_submit tagwait符合数量级，但不是证明全部请求进一个queue。CPU迁移、多queue映射、blockmerge、完成复用都需记录。16*1023是聚合容量，不能当单thread免费可用深度；相反burst>tags也不意味着每次必定阻塞，完成可在call内释放tags。

## 最便宜的因果实验（下一轮，不在本轮执行）

固定真实层/IDs、同一I/O实现/表/输出总量，ring4096不变，batch rows分128/256/512/1024，即reads256/512/1024/2048。此处只改submitburst，不改模型、缓存、队列depth、CPUaffinity或reader数。预热/随机交错成对重复，记录：

- 准备总T、syscall计数/耗时、accepted/completed；
- tag等待offcpu及内核stack；
- perhwqueue实际发出/completion分布；
- bytes和输出逐字节hash，allocation/copy成本；
- 最后同输入端到端GPU时间/TTFT/两rank就绪。

若burst下降使submitoffcpu/tagwait显著降且totalgather降：证实突发/队列匹配的软件机制可改善；若wait转移到ring/getevents且total不降，说明只是等待搬家，不能报收益。若total下降来自不同requestmerge/cache，则继续单变量分离。

batch512rows=>1024requests仍比1023tag多1，不能叫严格<=depth；若需严格界内应测试batch<=511或256。不要为了整齐假定1024等于1023。ring与burst两个变量不在同一首轮一起扫；第二轮再按服务完成率决定window。

如果队列tagwait分散在多个queue，先追实际devicecompletion瓶颈；增加reader可分散queue但同时改变并发/CPU/缓存与设备竞争，单独实验，不把聚合capacity当收益预测。driverdepth增大/换scheduler属于系统配置实验，风险更高，后置且设恢复，不先动。

## 研究边界

仍无真实GPUneed/host映射、worker同步、同输入observer对照或专家DRAM计算管线数据。Engram块层供给与host耦合已定位，但整体3500瓶颈未全部闭合。下一应从相同真实ID受控replay或预备诊断artifact改batch测因果，而非继续只读sysfs求唯一根因。

原数据nvme-queue-config.json，数学nvme-queue-account.json。未运行新请求、未改nr_requests/driver/scheduler。已知生产原容器此前持续运行，当前只读配置操作不会作为新性能验收证据。
