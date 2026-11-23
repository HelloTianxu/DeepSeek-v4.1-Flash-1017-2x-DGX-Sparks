# 无重启采样方案与工具门禁

2026-11-23，goal round4。仅工具能力/ELF读取，不attach生产，不改内核参数。

## 门禁事实

head已有perf、bpftrace、bpftool、gdb、strace，不需安装。perf_event_paranoid4、unprivileged_bpf_disabled2，tracefs读取无权限；sudo -n不可用。用户此前授权分析权限，但不得把密码写入命令/文件；若采用提权工具须走安全凭据输入机制，不通过持久脚本明文保存。当前未提权、未attach。

gdb/strace可能暂停线程/扰动生产，不作为第一选择。优先有界uprobes+map时标；需先在无生产对象的独立门禁验证权限与cleanup，再确定正式观测。若uprobe失败不转为全系统开perf或重载驱动。

## ELF地址与file offset已核实

live EXEC，text所在LOAD：file offset0x235280，VMA0x1245280，故text范围内file_offset=VMA−0x1010000。不能按第一个只读LOAD的0x1000000基址直接减；text与只读LOAD之间有0x10000差。下述file offset已按实际text LOAD计算，不能把VMA原值当偏移：

|事件|VMA|file offset|证据/作用|
|reader entry|0x1403e10|0x3f3e10|串行双层thread，记录reader tid与eh|
|engramAhead1 entry|0x1403f80|0x3f3f80|record eh=arg0,j=arg5,start=arg3,n=arg4（需核Zig ABI）|
|ready发布前|0x1404134|0x3f4134|strb ready；不直接猜此时callee参数寄存器仍保留|
|join之前|0x13fb500|0x3eb500|pthread_join callsite model1204|
|join返回后|0x13fb504|0x3eb504|调用返回后的第一指令|
|reader第二层调用|0x1403ea4|0x3f3ea4|验证串行时标|

这些偏移已由readelf PT_LOAD与反汇编核对，不构成动态采样结果。探针应绑定原image对应binary SHA及inode，先读取hash并核二进制变化，不跨升级重用偏移。

## 最小低事件率采样设计

- attach只过滤目标containerPID/TGID，EH标识匿名化，仅记录必要元数据；每thread维护engramAhead1入口eh/j/start/n与call depth，ready发布/函数exit从thread映射关联。错误退出与ready发布需区分，不能把exit当ready。
- join_enter记录调用线程与pthread handle；解析handle→reader tid不能靠猜；可用pthread_create/threadentry的验证映射或将join按EH/chunk关联。不能擅自从不确定struct布局读内存。
- data generation/request身份缺失就unverified，不能把同一chunk位置视为同一请求。reader start不等于GPU需求时刻，至少用已核GPU边界或对应GPU事件建立时间映射。
- BPF kernel monotonic与GPU/CUPTI timestamp不默认同源；先校准偏移/不确定度，否则仅判单机host事件先后，不能计算GPU暴露等待收益。两rank事件亦不直接相减。
- 不在49152次AIO每项上print；第一轮只低事件率reader/join/ready，每chunk几十事件。另轮采提交批统计/unique blocks，并估开销，避免observer改变I/O供给。
- 输出缓冲有界与lost-event计数；完成后确认探针卸载、maps关闭、进程树为空、无残留profile；持久证据保留。退出/timeout同清理路径。

## 建议正式窗口

方案A（优先）：无重启，最长30秒有界uprobe采样覆盖少量同长度prefill；请求本身<60s，时标及工具门禁先确认。附加探针仍是生产观测行为，不能仅称“只读零影响”；先静态fixture与独立权限/开销门禁。可能无需10–15分钟重启窗口，但未验证。

方案B（fallback）：独立诊断artifact后10–15分钟受控重启，完整host+GPU事件同源。若预期构建更久另报时长，不夹在窗口里。

当前局限：尚无权限门禁成功的动态probe；没有measured j0/j1ready/join事件。现有analyze_engram_events.py 12tests能检算术/时钟/generation，但真实instrument接口还须严格适配，不能用synthetic结果下生产判决。
