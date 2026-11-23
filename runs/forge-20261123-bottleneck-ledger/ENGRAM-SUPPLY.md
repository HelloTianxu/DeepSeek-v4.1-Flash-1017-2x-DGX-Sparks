# Engram供给结构：请求放大与跨层等待耦合

2026-11-23，复用已有trace/源码、只读当前容器，不重启、不跑GPU实验、不下载。

## 已核实与计算

live TF_DS_ENGRAM_AIO=1、TF_DS_ENGRAM_LEAD未设置、interleave0。实际Engram两表weight行256bytes、scale行8bytes；投影K6144/256=24列，TP2每rank12列。每2048token、每rank、每Engram层需24576行。

参考engram_aio.submitLocked每行单独weight+scale的4KiB对齐读取，无dedup；因此49152requests、最低201326592请求bytes、有效6488064bytes，最低放大31.03×。AIO4096slots，aioGather每批1024行，共24batches和12ringlaps。绝对文件偏移可能导致少量8KiB读取，所以此值为请求下界，非SSD物理读量。

当前进程已经打开两Engram表各一个O_DIRECT fd，支持真实direct路径的前提。最初从host读取/proc root-owned进程fd被PermissionError拒绝；随后在现有容器root只读成功。第一次错误硬编码O_DIRECT=0o40000（x86常量）误标false，保留engram-open-fds.json负证据；以远端os.O_DIRECT=0o200000（arm64）修正，engram-open-fds-native.json确认119/121direct=true。未保存/使用密码，未修改系统权限。

## 新的代码级候选：第1层等待可能还在等第14层

model.fillChunk第14层完成时启动engramReadAheadAll，依次读取下一chunk两个Engram层；下一chunk进入时取m.ahead并在第1层做整个thread.join，之后才检查ahead_ready/engramApply。即使下一chunk第1层ready，thread仍在读第14层，也可能被join阻塞。单thread双层、单paio ring形成跨层就绪耦合。TF_DS_ENGRAM_LEAD独立层reader/ring可解耦，但当前未启；不能直接宣称启用可回收0.664s。

已有trace第2请求前chunk第14层Engram GPU完成到下chunk第1层空隙起点约310.0/312.4/311.1ms；再加gap161.8/153.0/132.6ms。GPU时间不是reader启动时间，不能把这两段之和当真实I/O时间；足以指出“可能有约0.31s可重叠机会，却仍在第1层等”必须用host时标验证。

替代解释：reader从未提前启动（fill_seq长度/指针条件或piece界限）、启动了但first layer本身慢、first已ready而second慢、CPU解码/reader调度慢。不得用源码存在预取证明预取实际发生。

## 最小判别字段与预测

每chunk记录t_start_reader、j0ready、j1ready、join_enter/exit、ahead_start/n、fill_seq.len/ptr_match、hash/read/decode分段。不在计时内同步GPU。用固定请求双rank对齐：

- 若j0ready<join_enter<j1ready≈join_exit：证实跨层join耦合是关键等待。
- 若reader在本chunk才start：证实预取机会未实现，追条件/调度。
- 若j0ready≈join_exit且read主耗时：first层供给受读路径限制。
- 若read先完成decode晚：CPU解码供给；不可用drive带宽解释。

同步采真实I/O提交bytes与completion/reap、唯一4KiB块和duplicate率。31×是请求格式放大，不自动等于31×可加速；同请求重复token/块会使dedup成为软件机会，但实际duplicate率未测。SSD/controller可合并或缓存，O_DIRECT仅绕host页缓存。

将49152除gap会得0.227–0.371M requests/s，仅是“所有请求都在gap内”的条件匹配，不是测得IOPS；真实有overlap，严禁把它当硬件饱和证据。

文件：engram-table-metadata.json、engram-live-mode.txt、engram-open-fds-native.json、engram-lead-windows.json、engram-io-account.json。新成果是把Engram问题从泛I/O缩至请求放大+双层join耦合的可证伪候选，没有优化上线或最终根因证明。
