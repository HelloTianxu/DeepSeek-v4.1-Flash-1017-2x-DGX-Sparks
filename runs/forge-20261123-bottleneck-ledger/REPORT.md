# 双Spark prefill瓶颈账：位置已识别，主内核机制未唯一识别

日期：2026-11-23，Asia/Shanghai。只读复用生产配置、checkpoint safetensors头和已有trace；未重启、未下载、未运行模型/GPU实验，未使用或保存密码。目标3500指prefill输入token/s，不是生成token/s。

## 1. 判决

【工程实测】稳定制约来自：专家核心约1.91s；其他GPU工作约2.39s（含collective、变换、HC、注意力等）；head无kernel间隙约0.67s。其中0.664s集中于四个完整chunk的第1层Engram入口，约占空隙99%。不能把整个负载定为硬件算力已满，也不能把全部耗时归为软件缺陷。

【研究假设】专家内核受权重复用/带宽、融合解码指令供给、低驻留warp及小专家组形状共同制约；这些机制都能解释现有部分观测，尚未唯一识别。Engram供给/预取未覆盖的边界已定位，但SSD读取、哈希、解码、peer等待/同步的因果贡献尚未分离。

## 2. 账面必须先校准

live模型：EXL3 EAQ v1.1 2.9bpw+ablit，D5120、I2304、40层、384 routed+1 shared、每token6 routed+1 shared；真实内核HMMA FP16输入/FP32累加，不可用1PFLOP sparseFP4作为分母。

trace所有kernel在head stream13。第2请求8628输入token、window4.972653679s（server约4.977s），已预热library load0。观察到4×20次R2048、20次R436、20次R128专家调用，合120次。源码model.fillChunk在li20执行replayCut，prompt.replayCut将decoder裁到窗口128行，与观测闭合：

`token_layers = 20×8628 + 20×128 = 175120`

不是`40×8628=345120`。因此按完整40层计算有效工作会夸大1.971×。

【理论推导，依赖TP2布局与参考ABI】head本地I=1152（gateup grid.y18×64），down D5120，单配对有效矩阵FLOP为`6×5120×1152=35389440`。总pairs=175120×7=1225840；head专家有效FLOP=43.3818TFLOP，两rank总有效FLOP=86.7636TFLOP。head核心1.908546s，等效22.73TFLOPS。此数字不含padding、解码、旋转，不是硬件利用率；worker未有完整同步trace，不能宣称两rank合计45.46TFLOPS实测。

## 3. 稳态时间与Amdahl边界

第2请求head kernel union4.301959s，no-kernel0.670695s。core sum与union均1.908546s。分类累计（除16.293ms部分重叠外闭合）：

|分类|累计秒|占window|
|---|---:|---:|
|专家gate/up+down|1.908546|38.38%|
|collective kernel|0.538977|10.84%|
|其他GEMM（稠密、路由等未进一步拆）|0.503544|10.13%|
|HC混合|0.398465|8.01%|
|attention名称内核|0.258425|5.20%|
|rotation（专家与稠密混合）|0.230837|4.64%|
|专家epilogue/combine|0.197363|3.97%|
|unpack（主要稠密路径，非全归专家）|0.098744|1.99%|
|其他GPU|0.183340|3.69%|
|类别重叠抵消|−0.016293|−0.33%|
|无kernel间隙|0.670695|13.49%|

collective包含传输与peer等待，不是纯网络搬运成本。attention名称分类不是完整attention子系统。

目标预算`8628/3500=2.465143s`，当前需回收2.507511s。若只消除专家core且其余保持不变：`4.972654−1.908546=3.064108s`，上限2815.83tps。即使消掉全部head no-kernel间隙：上限2005.60tps。二者都归零的极端反事实还剩2.393413s，才略过3500。实际内核不可能归零，因此必须同时改进多个实质工作项，不能只靠“调大chunk”或“修上传”。这些是固定其他时间的反事实，不是软件收益上界：调度改变可能同时影响其他项。

第3请求重复：T4.958350s、core1.923964s、idle0.668079s，结论稳定。所有结果在profile环境，不等于无profile正式速度验收。

## 4. 形状曲线：实测相关性与贡献量级

第2请求配对gateup/down后的中位数（同层配对后再取中位数，不是两个median盲加）：

|R|总pairs|core中位ms|有效TFLOPS（参考TP2）|累计core ms|
|---:|---:|---:|---:|---:|
|2048|14336|20.1620|约25.16|1607.0（80次）|
|436|3052|9.7333|约11.10|200.9（20次）|
|128|896|4.9725|约6.38|100.6（20次）|

总pairs不是单专家GEMM的M；真实M_e=count[e]，共享专家M_shared=R；routed总计6R。总pairs16倍而耗时约4倍同时符合计算+读取摊销模型、padding与缓存模型，不能独凭它判为软件缺陷。

条件外推：将所有小形状按80个bulk调用的加权单位pair成本归一，投影节约190.9ms（window3.84%），不是实测收益、不是上限。尾块按相同encoder层的bulk单位成本归一，差115.4ms；仍有位置/路由混淆。不能把小形状4倍单位效率损失乘到整个请求。主要core时间1607ms在bulk，研究bulk执行效率比仅修尾块更重要。

## 5. 带宽匹配：排除错误分母，保留竞争解释

【工程实测，checkpoint头】逐层完整385专家唯一序列化字节：多数5.131–5.135GB；第18–22层3.432GB。前20层平均完整5.0GB，按TP2均分的条件估计约2.482GB/rank/layer，后20约2.439GB。它不是实际rank-cache字节，更不是DRAM traffic。

单Spark官方带宽273GB/s。大形状core中位20.162ms若完全带宽饱和，对应`273×0.020162=5.504GB`，约为所有专家唯一半份权重2.2倍。考虑多tile、gateup/down不同tile尺寸、重复读取与L2，此数量级完全可能；故不能因为一次唯一权重理论只需9.1ms而直接定算力瓶颈。反之不能拿唯一字节/time≈126GB/s当实际带宽利用率。

尾块9.733ms在273GB/s下对应2.657GB，接近所有专家唯一半份约2.482GB，因此读取摊销不足是合理竞争假设。R128在273GB/s下只能搬1.358GB，低于所有专家半份约2.439GB，说明“每调用全部专家权重都从DRAM读一遍”与数据不相容；需要路由子集、不同rank布局或缓存复用。不存在统一8GB/token读取账可直接用于当前prefill。

严格屋顶线应写为：`Tcore >= max(Fexecuted/Csustained, Bdram/BWsustained, Idecode/Rdecode)`；现有只知道F有效、唯一序列化字节、静态指令与时间，不知道实际执行F、DRAM字节、动态解码指令和可持续吞吐，系统未识别。不能从少量点数学反演唯一根因。

## 6. 占用率资源匹配

【工程实测CUDA属性】GB10：48SM，1536threads/SM（48warps），65536regs/SM，102400shared bytes/SM。

【理论上界】gateup106regs×256threads=27136regs/block，44288shared/block；resident blocks≤min(2,2,6)=2，即warps≤16/48=33.3%。down255regs×128=32640，32896shared；blocks≤min(2,3,12)=2，即8/48=16.7%。分配粒度或其他资源只会进一步收紧。这是驻留上界，不是测得occupancy或吞吐上界；少warp也可饱和特定管线，不能预测“降低寄存器可提3倍/6倍”。资源事实使延迟隐藏不足成为有依据候选。

## 7. 等待边界：已缩小到Engram入口

第2请求四个>10ms空隙216.106/161.824/152.981/132.638ms，合663.549ms，占no-kernel98.93%。每次在该chunk第0层MoE及collective之后、第1层Engram前；下一第6个kernel为Engram名称内核，源码li1主机reader join/lead.before后才engramApply，与边界吻合。第3请求四段合660.781ms，同结构。

【工程实测】HtoD API2.661657s中2.656591s与GPU kernel重叠（99.81%），只5.066ms与no-kernel重叠；所有CUDA API合并与idle重叠仅5.951ms。故不能把2.66s“主机复制等待”直接当关键路径浪费，也不应继续把pinned4KB微测试当主诊断。kernel launch API合约46.49ms，99%与GPU重叠，不支持以算子launch主机开销为当前大瓶颈。

【研究假设】Engram前供给可能包含CPU哈希/解码、SSD缺页/I/O、预取启动时机、线程join、rank/peer依赖。边界不是因果标签：线程栈/文件I/O/worker timeline未测，不能报告SSD造成0.664s。此处最便宜的判别证据：每chunk每Engram的hash/read/decode起止、ready/join阻塞、I/O字节及两rank同步；不改数学执行。

## 8. 研究下一步（优先定位，不先优化）

1. 首先为上述四段Engram供给做CPU阶段时标及reader就绪时标，对齐同一请求GPU时间；区分供给慢、启动晚、peer等待。它可能回收约0.66s量级，但单独不够3500。
2. 同时选固定encoder层bulk专家调用，采counts/有效tile并用真实量化权重原样重放，DRAM/L2字节、tensor/ALU/long-scoreboard及occupancy。只读已核实驱动`RmProfilingAdminOnly:1`，生产容器仅增加CAP_IPC_LOCK，没有SYS_ADMIN，符合此前计数器拒绝的权限机制。用户已授权分析权限，后续可仅在隔离容器添加必要能力验证，不必全系统开放计数器或重载驱动；本轮未创建该容器、未改权限。
3. 没有计数器时做固定层/路由/总pair的切分对照，独立控制缓存冷暖和rank；改变tile/reuse与解码路径需分别消融。普通BF16GEMM不能替代真实融合EXL3内核屋顶。
4. worker采集缺口必须补齐才能分离collective0.54s中纯链路与peer等待。只读已核实head enp1s0f1np1速率200000Mb/s、MTU1500，即单rail线速25GB/s。0.539s若纯线速搬运对应13.475GB；实际collective字节未测且双rail配置不等于实效合并，不能据此报网络利用率。需要读取实际消息字节与两rank就绪时标，不把聚合理论带宽直接换成单流收益。

本轮没有执行受控重启。完整采样预计需要原先提出的10–15分钟窗口，用户本轮要求继续定位不等于确认该窗口；优先当前只读分析。计数器、驻留、tile是判别实验，不是通过权限即可立即得到根因。

## 9. 复现与证据

运行`python3 calculate_ledger.py`、`python3 reconcile_layers.py`、`python3 wait_ledger.py`，结果ledger.json/layer-shape-ledger.json/wait-ledger.json。gap-neighbors.json为四段前后kernel。checkpoint-config.json/all-layer-expert-storage.json为本地模型元数据；cuda-device-attributes.json为不执行kernel的CUDA属性查询结果。原trace来源前轮trace.sqlite与timeline.json，保留原数据，不覆写生产文件。

局限：没有worker全timeline；无同期持续带宽、同格式同形状饱和基准；没有真实count、动态指令计数；R/li通过trace几何与顺序推断，非实参/NVTX层标；正式无profile成绩、数值等价与输出质量另需验收。所有不足已在账中保留，未伪造实测。
