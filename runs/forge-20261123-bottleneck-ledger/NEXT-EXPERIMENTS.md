# 下一轮验证与改进计划：先归因，再补木桶

2026-11-23，Asia/Shanghai。此文为实验计划，非已执行结果。依据REPORT.md与ENGRAM-SUPPLY.md。目标仍是先识别瓶颈；3500为prefill输入token/s，非生成速度。

## 0. 决策与资源边界

优先E1 Engram关键路径归因，再E2真实专家bulk内核归因；通信就绪时标随E1一起采，不另起泛化网络测试。所有源码/工具/权重先复用本地。生产启动参数、模型、路由k、KV语义不变。未经新的时间窗口确认，不重启、不运行预计>60s任务。当前只提交设计，不宣称实验已经运行。

每项先验证源码与live二进制构建身份；reference源码不是live执行证明。任何插桩在独立实验工程中制作，不覆写生产镜像或原启动脚本。长编译/加载时长须先估算并确认，不能藏在10–15分钟采样窗口内。

## 1. 假设与判决矩阵

|编号|可证伪假设|必要观测|支持判据|证伪/降级判据|下一改进|
|---|---|---|---|---|---|
|H1|当前层ready但whole-thread join等待后层|j0/j1发布ready、join enter/exit、GPU依赖|j0ready在join前，j1ready晚，退出跟随j1；对齐真正GPU需求时刻|j0本身晚，或没有提前reader|层级ready同步，独立读者/环，保留错误与代际保护|
|H2|预取未启动或启动太晚|reader start、fill_seq长度/同源标识、chunk/piece状态|供给启动未利用可用依赖窗口，且本chunk才启动|提前启动充分但处理仍未完成|修启动条件/调度，不能绕过prompt身份安全检查|
|H3|随机读请求/完成回收是供给主成本|hash/read/decode、submit/reap、请求bytes、唯一块|read占主要供给耗时且占关键等待；重复块可减少实际提交|CPU解码主导或读已被隐藏|先行/块去重及scatter，后考虑存储布局重排|
|H4|CPU hash/decode供给不足|CPU阶段时标、线程运行/等待|read完成早、decode/调度完成晚，持续形成关键等待|CPU阶段短且不在关键路径|仅优化被测阶段，避免与GPU争共享带宽|
|H5|专家bulk主要受DRAM读取限制|实际DRAM/L2字节、同形状可持续带宽、Tensor/ALU/stall|Bdram/BWsustained可解释时间，减少重复读取独立对照有效|低带宽且其他管线饱和或改变流量不改时间|布局/分组/权重复用；不是换量化格式|
|H6|专家bulk主要受融合解码/执行供给限制|动态ALU/Tensor、stall、执行算术、资源|内存账解释不了时间，解码减少/复用对照有效|减少解码无效且访存主导|解码复用、tile/流水深度单变量改造|
|H7|collective主要是等对端而非链路|两rank相同消息的ready/start/end、实际字节|等待随peer就绪延迟；自身搬运账明显较小|双方同步ready仍传输占主时长|先解供给失衡；纯链路证实后再调传输|

判据不是看到一个counter百分比就定罪。保留并存瓶颈；一次消融只改变一个机制。无需强迫每个内核拥有唯一瓶颈。

## 2. E1：一次诊断采样区分Engram四种原因

### 采样字段

记录request匿名ID、rank、chunk起止、layer、generation、reader启动与结束；hash/read/decode起止；j0/j1ready发布时刻；join enter/exit；ready是否命中、其chunk/generation；fill_seq长度与同源布尔值，不记录原prompt或裸指针。记录每层AIO submit/reap请求数与字节、完整4KiB跨度集合及重复率。不能仅按起始块去重漏掉8KiB跨块请求。

记录两rank消息序号/字节数及就绪/完成时间，同步GPU事件/NVTX；单机用同一单调时钟，跨机先建立时钟偏移与不确定度，不能用未经校准的两个时间戳相减。无需上传原输入文本。

### 不污染性能

线程本地预分配固定大小记录区；计时内不printf、不JSON、不新增GPU同步，不新增逐请求哈希集合。提交信息先记录到有界缓冲，唯一块统计在采样后离线算。ready时标不得用无同步读取bool：记录线程自己的发布事件并在offline合并，避免data race。记录overflow与时钟误差，缺失事件不能算证实。

先比较同镜像插桩关闭/开启，固定cache与warm状态交错采样。若插桩效应超重复噪声，重新缩小字段，不用有偏trace预测收益。

### 样本

复用已有固定corpus/token IDs（代码、自然语言各至少一类），缓存命中0且相同token长度；相同prompt可重放但必须显式清理/禁用prefix命中。区分首次初始化、暖内核冷前缀、Engram存储缓存状态。固定单流，记录并发空闲、时钟/温度/功耗和后台I/O。记录真实N，不用字符串长度代替token。

### 关键量

令d0为GPU实际需要第1层数据的时刻，r0/r1为层级ready，re为整线程结束：

- 当前暴露等待约max(0,re−d0)。
- 层级同步的理论等待约max(0,r0−d0)。
- 仅解除跨层join的条件可回收量为上述二者之差；忽略改造自身成本时才成立。

不能把CPU join时长直接当GPU idle；CPU可在GPU忙时等待。reader先后及两rankcollective依赖一起判断。首chunk没有前chunk供给窗口，单独统计，不能假定全部0.664s可被提前读取覆盖。

## 3. E1改进分支与无损约束

A. H1成立：先复用已有engram-lead机制做独立实验，或最小层级发布/等待实现。只有ready acquire/release或明确同步、prompt generation匹配、buffer不被下一请求覆盖、取消/错误/两rank fault传播、join与清理完整，才可进入性能测量。不能直接在整线程未结束时复用其hash/ring/buffers。已有第二ring4096×8192=32MiB为环增量；另有hash/状态/buffer增量，需从实际分配算总预算，不拿32MiB当总内存。

B. H2成立：修复预取启动时机/合法条件，保留跨prompt安全检查；不可“去掉join”掩盖数据生命周期问题。

C. H3成立且重复块多：先做单次batch的按(file身份,块偏移,长度)去重，读一次scatter回原顺序，输出应逐字节一致。唯一块数U、原请求数Q，Q/U只是请求数比，不是速度比；需计入排序、scatter、额外内存和不同page读取范围。随后才评估离线交错weight/scale布局；布局改变必须完整验证所有索引/offset。

D. H4成立：固定read结果独立测hash/decode路径；仅优化该阶段并测供给关键路径，不把CPU闲置自动等价可提速。

所有分支先单独AB/BA，再组合，避免同一0.664s重复计收益。

## 4. E2：真实专家bulk，不替换成普通GEMM

选同一encoder层的R2048调用，锁定权重布局、codebook/k2、input/pick/weights、slots、本地D/I及dispatch。捕获counts和有效work项，区分有效F与padding后的executed F。全模型不得重复加载；单层/必要权重子集的峰值预算在运行前核算。原样replay先对输出与时间，replay不匹配生产就暂停归因。

先在隔离容器最小能力范围验证计数器可读（已有RmProfilingAdminOnly1，原容器无SYS_ADMIN），不改全系统driver策略。计数器replay必须仅限隔离调用；不得production multi-pass。使用工具实际列出的metric，不写未经本版本确认的名字。

最小对照：固定总有效pair与路由序列，R128/512/1024/2048切分；每点记录calls、实际tiles、L2/DRAM bytes、Tensor/ALU、long-scoreboard与resident warps。冷/暖cache分别成组，不能强迫清系统缓存。可加4096仅在scratch/权重/设备限制通过后。切分可能改变dispatch与浮点实现，全部记录，不能视为同内核纯M效应。

后续解码复用/减少读取/tile配置分别单变量；先同布局真实路径可持续上限，再选改善。低occupancy或高ALU计数不能独自证明瓶颈。

## 5. 木桶收益账：只加独立关键路径时间差，不连乘

以req2 profile窗口为条件参考：T0=4.972654s，N8628，专家core1.908546s；Engram入口idle0.663549s。以下不是预测或承诺，r为core耗时降低比例，g为暴露等待实际回收秒：

`T_candidate=T0−g−r×1.908546`，需其他项/重叠不变才可用。主机/API累计、raw I/O时间不进入此式。

|g 秒|core降低 r|条件T秒|条件prefill tok/s|
|---:|---:|---:|---:|
|0.30|0|4.673|1846|
|0.60|0|4.373|1973|
|0.30|25%|4.196|2056|
|0.60|25%|3.896|2215|
|0.60|50%|3.418|2524|

即便g0.60且core减半，离3500预算2.465s仍差约0.953s，必须从其他GEMM/HC/变换/collective等找额外实测收益。若核心减半+Engram回收后瓶颈迁移，更新账而不延用旧占比。改进后一律重算critical path与测总T；模型只是排优先级，不替代成片性能数据。

## 6. 性能与正确性门禁

- 本轮机制改造不改k/量化/KV数学，目标无新增质量损失；不把相同argmax或短hash当分布无损证明。
- I/O/调度改造要求Engram rows逐字节一致、无错prompt代际、tensor输出与同配置基线一致；固定seed输出与logits检查，非greedy亦纳入。若浮点路径变化，先声明数值差异并验证分布/质量，不自行放宽一期无损门禁。
- 固定token输入，AB/BA交错重复；先少量预跑估噪声与时长，再确定样本数。每次保留原始duration，报告配对差、区间和异常，不只最优值。
- 进入改进阶段的建议门槛：端到端差的置信区间下界>0，且超过插桩/运行漂移；优先≥3%有复现收益，<3%保留负/小结果再定投入。此为研究筛选，不变更用户目标。
- 正式验收从无profiler生产同配置测prefill、TTFT、输出平滑、P99、能耗；保持既定1K/8K TTFT门禁，不把prefill吞吐代替TTFT。
- 取消、错误、连续不同prompt、prefix恢复/同position不同prompt、两rank错读fault路径与压力/内存门禁必须过。

## 7. 执行顺序与窗口

1. 当前可做：源码身份核对、插桩设计、分析器/fixture与预算；不跑模型、不扰动生产。预计>60s的构建另报时长。
2. E1第一窗口：预计10–15分钟，至多诊断入口切换与原服务恢复两次重启；仅采baseline，不夹带LEAD开关和I/O改造。窗口前独立artifact已就绪，锁、disk、headroom检查通过。
3. E1归因报告：每rank每chunk供给时间线、暴露等待分解、I/O请求/唯一块、首chunkvs后续，给明确H1–H4 verdict或unverified。
4. 根据verdict选择一个E1改进，独立预计时间并确认新窗口；不可用诊断批准覆盖后续无限实验。
5. E2隔离真实replay和counter，时间/容量单独估算；之后才优化bulk。通信时标复用E1数据先判H7。

## 8. 回退与最终交付

执行前固化原镜像ID、启动脚本/config hash、units/container参数、baseline输入输出证据、实验唯一名；只切独立入口。trap/finally超时均恢复原入口、停整个实验进程树/容器、删除临时dropin/wrapper、释放lock/port，并查两rank健康、无profilers/stragglers、配置hash与原输出。原研究证据非临时，保留不删。任何无法清除项点名，不碰非本实验进程。

每实验交付5项：假设与账面、精确配置/commit/复现命令、原始时标/输出/性能数据、单变量归因与噪声、回退验证与下一判决。失败/收益为零同样入research-log。

当前状态：计划已设计；未采j0/j1ready、未证跨层join、未得到实际DRAM traffic、未上线任何优化。目标保持active，下一步需要时间窗口确认。
