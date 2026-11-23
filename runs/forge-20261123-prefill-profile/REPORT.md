# Prefill瓶颈定位第一阶段：干净分项剖析

日期2026-11-23，目标3500tok/s。用户批准15–20min最多2次重启，恢复优先。已执行两次服务循环，原local.sh SHA256逐字节恢复为cb795ef2db83fd303d9448381842cbe339a4fb4d538261373019a9ba7a07351a。原镜像8348338a8d90、权重、时钟、spill、memory guards均未改。

## 已测结论

仅profile2、max_tokens1、无decode verify，follower rounds0，无CUDA_ERROR/follow_refused。两条冷技术文档请求16702/8534tokens，cached0。profile在正常退出后输出14chunks聚合阶段均值。正常前对照约16K服务端1785.5tok/s；profile约1706.7；恢复后带采样1707.7，另一次带采样1780.7。输入nonce、token数及缓存温度略不同，不能把差异全部归因profile。

| 桶 | 每chunk聚合均值ms | 占已记录桶和 |
|---|---:|---:|
| MoE |486.123|46.27%|
| attention |203.227|19.34%|
| Engram |186.968|17.80%|
| gather_a+gather_m |95.066|9.05%|
| mix_attn+mix_ffn |78.640|7.49%|
| embed+head |0.590|0.06%|
| 合计 |1050.614|100%|

证据rank1-profile.txt:41–43、phase-summary.json。桶为GPU event间隔，包含CPU未及时提交造成的GPU空窗、网络/同步等等待，不能称独立纯kernel计算。MoE包含routing/EXL3专家/潜在2D内部交换；attention尚未拆indexer/projections/core；Engram包含read-ahead join之后应用阶段和前置等待，内部rows0不意味着SSD零开销。末尾drafter absorb未完全覆盖。partial tail chunks与不同请求混合，不能按一个完整2048chunk推导总速度。需要下一轮kernel-level timeline/消融才能得到根因。

## 新瓶颈地图

首要桶MoE；次要attention；Engram与交换也不可忽略。没有单一百分比证明compute-bound，更无CPU算力饱和证据。仅MoE提2x在这个分桶账上最多约1/(1-.4627/2)=1.30x；从当前约1780到3500需近49%端到端降时，不能只把一段加速2x就宣称完成。桶权重受instrumentation影响，此Amdahl仅条件推导。

## 资源采样

telemetry.jsonl首次因/proc/container PID io权限拒绝失败，保留，不声称有效。telemetry-normal.jsonl采样未与请求重叠，GPU0%，是空闲证据，不能用于负载归因。telemetry-overlap.jsonl通过父进程同时发probe与采样，确有GPU96%样本：两机CPU全核平均时间窗口约8–11%busy，NVMe读取约200–400MB/s；从系统diskstats而非进程I/O推导。样本稀疏，跨部分空闲边界、系统全局I/O且含采样开销，不能证明单线程无瓶颈或某请求I/O量。sampler等待15秒超时后父finally terminate并确认本机无telemetry.py残留；保留partial有效样本。RDMA计数器不能把0值直接解释成prefill无通信，prefill可能走NCCL而非decode RDMA ring。

## 上轮更正

旧profile-rank1.txt末尾其实有prefill分桶，先前遗漏，已主动撤回“无任何profile”。decode错误来自eager timer进入CUDAgraph的代码机制匹配，不把旧decode counters当成功rounds。此次prefill-only路线已实测无错误。

## 回退与质量

本轮只在remote config追加RANK1_PROFILE2；正常退出收日志后，核对backup与当前预期内容，恢复完整原文，SHA相同，再user systemd启动。cleanup.json两rankrunning、image相同、profile缺失、最近log无CUDA errors。恢复后code512tokenshash1428664efda9、tool47hash310581e3ca24同原基线。未证明任意输入分布等价但没有数值代码/权重修改。两个probe结果以及SSE/metrics保存在相邻forge-20261123-interleave的PBASE/PONLY/PAFTER/POVERLAP JSON；不重写原始数据。

## 下一步最小可证伪实验，尚未执行

1. 先捕获两rank短kernel/API timeline，拆MoE的group/rot_in/gateup/down/combine与内部exchange。本机已装nsys/ncu，不下载；使用可回滚wrapper或独立受控lane，不改正式image。
2. 分开量weight解码指令、FP16mma/scalar路径、实际DRAM/L2字节与有效带宽，再决定compute或memory屋顶线；仅GPU96%并不证明Tensor饱和。
3. Attention拆投影/indexer/core；以相同tokenized输入/路由重放归因。
4. Engram读前置与等待用trace标出，不把pagecache冷prompt与冷disk混同；不drop_caches污染生产。
5. 量两机不平衡与每层交换，按max(rank0,rank1)+不可隐藏通信建依赖账；先定位再改布局/分块/算子。

本阶段定位到大桶而不是最终微观根因。3500尚未达到，继续任务需要下一维护窗口，生产服务现已恢复。
