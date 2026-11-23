# 动态系统调用归因：io_submit占gather84.88%

2026-11-23，goal round9。现有tracepoint字段先用bpftrace -lv核实ctx_id/nr/ret，隔离--rm容器清理成功。18probe20秒同步SUBMITSYNC1请求N8583cached0prefill4.962337417s，hash528b0f3da89f；未重启或改变优化参数。

8完整层gather1893.165587ms，start1831.848836ms，io_submit1606.945600ms；submit/gather84.8814%，submit/start87.7226%。单层24syscalls，49152requested/accepted，0nonpositive return；tail391row5syscalls，9384requested/accepted。真实请求数验证24rows/token（12行*weightscale2），非仅reference计算。

主要耗时发生在io_submit期间，纯用户态start扣syscall约224.903236ms（跨8层），不能把全部start96%归ring用户态轮询。io_submit包括内核构造/块层driver提交、资源/完成背压、调度/睡眠，尚未区分CPU工作与阻塞。零错误不等于无内核背压，accepted不等于completed。

本次bpftrace signed/unsigned警告保留：nr为有符号但实际均正，ret>0才累加；accepted==requested==24*n逐层assert通过。下一版应显式正值检查与uint64cast消除警告，无需为此重复当前实验以追漂亮数字。

没有对高频io_getevents/reap逐次挂probe，以避免busyloopobserver放大；下一步低率perf/sched采样：仅io_submit active的reader tid，计running/offcpu与kernel stack，分辨内核CPU开销、等待块层/设备资源与其他调度。须先核kernel符号可见与profile能力、lostevents/cleanup；不能用kernel栈样本自动等于时间占比。正式GPU/host映射与worker供给依然未采，不能从总T直接扣1.607s。

证据trace_engram_submit.py、engram-submit-trace.txt、engram-submit-summary.json与SUBMITSYNC1请求JSON。各命令<=55秒，BPFinterval20/timeout30，CLEANUP_ABSENT True。数据为诊断非性能改进成绩，目标继续active。
