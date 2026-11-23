# DGX Spark 第一轮受控调优报告

日期：2026-11-23。证据：本目录原始 JSON、SSE 时间戳、两机配置快照和日志；不采用此前会话中未归档的基线。

## 结论先行

1. 已配置 `PREFILL_INTERLEAVE_MS=0` 并显式传入 `TF_DS_PREFILL_INTERLEAVE_MS=0`。A0→B1→A1→B2 的正常单请求对照没有观察到 prefill 增益，不能称作加速落地。0 的并发 decode 停顿代价尚未量化。
2. 在本次自然技术文档负载下，服务端冷 prefill 8K约1700、16K约1750–1780 tok/s；3000 未达到。用户端 TTFT 口径略低，分别另报，不能混用。
3. 默认草稿调度下，自然代码512-token请求约83、长文约61 tok/s。200+ 未达到；未从 SSE burst 虚构滚动峰值。
4. `TF_DS_FIX_K=5` 实际增加了验证候选，但拖慢代码与长文，已移除；未修改模型、量化或任何数值内核。
5. `RANK1_PROFILE=2` 产生 CUDA event invalid-value/follower refused，剖析结果不可用。已停掉该环境、移除profile并恢复正常服务，不把其计时当作瓶颈证据。

## 七项立项检查

- 现象：提高冷prompt处理速度与单用户连续输出速度，不追求多流aggregate。
- 物理：prefill需要压总时间；decode需要压轮次成本T或提每轮有效输出τ。
- 假设：Mac wired拐点、缓存token占比、频率读数均不证明GB10对应瓶颈。
- 不对称：固定checkpoint、固定两机和本地已有源码/镜像允许模型专属消融。
- 理想终态：权重常驻；只做能兑现的候选；无不必要同步、重复计算或I/O等待。
- 最小证伪：interleave无竞争负对照；FIX_K候选变多而τ/T变差。
- 回退：只保留用户指定的0；恢复默认allocator与无profile状态，系统守卫不变。

## 一、环境和安全

两台Spark：spark-bada与spark-bbf1；TP2、双RoCE rail、当前镜像 `tensorfold-dsv41:exl3-tp2-8348338a8d90`；EAQ v1.1 2.9bpw + abliterated overlay；context524288、pool1052672、spill64GiB、KV shard on。这些是现有参考模型，不是未量化原模型。

服务由head user systemd管理，直接start.sh会持锁失败。维护用现有systemd，不杀外部进程、不删除锁。连续维护启动触及StartLimitBurst=3/30min，使用reset-failed恢复，不改unit限制。所有文件/镜像/模型复用，无下载、构建或大页/时钟/内存参数实验。

原启动配置见deployment-before；远端原配置backup SHA256：c8bd78f42a17611895952ca44be5d5a3054d7f74ccda47aac68f37e6038c1b60。内存守卫、reserve、认证、spill内容保持。

## 二、prefill单变量消融【工程实测】

数据精确值见prefill-summary.json。每条cached_tokens=0，thinking off、max_tokens1。同主体自然技术文档，arm nonce置于位置0，故token数略有差异；按实际token数归一化。单组不足置信区间，结论限定为本次样本。

| arm | interleave | 约8K 服务端 tok/s | 约16K 服务端 tok/s | 约16K客户端 tok/s |
|---|---:|---:|---:|---:|
| A0 | 250 | 1724.6 | 1771.5 | 1752.3 |
| B1 | 0 | 1717.4 | 1780.0 | 1757.1 |
| A1 | 250 | 1714.8 | 1775.9 | 1745.0 |
| B2 | 0 | 1700.0 | 1750.2 | 1736.1 |

服务端速度=usage.prompt_tokens/server.prefill_seconds；客户端=(prompt-cached)/首次非空输出到达时间。前者亦不是硬件计数器意义上的纯GPU时间。

B1 rep1匹配长prompt重启后从spill恢复8192/16384token，保留原始数据但排除。重启不能清除磁盘prefix cache；没有删除它来造冷态。A0 rep0为较短pilot，排除。

代码审查说明：interleave仅有其他runnable decode时工作；不是每chunk无条件250ms睡眠。SPILL pending服务仍可独立执行。未做竞争阳性实验，因此不能报告并发净收益。

## 三、decode的τ/T账【工程实测+理论推导】

下表为完整512-token请求，不是瞬时burst。rounds来自rank0原始日志；accepted/drafted来自请求隔离前后metrics且与日志相符。

| 配置/任务 | 实际轮数 | 接受/候选 | 每轮可见输出(511/rounds) | 平均轮次ms | 服务端decode tok/s |
|---|---:|---:|---:|---:|---:|
| 默认，代码 | 153 | 358/529 | 3.340 | 40.170 | 83.14 |
| FIX_K5，代码 | 145 | 366/723 | 3.524 | 46.590 | 75.64 |
| 默认，长文 | 257 | 254/420 | 1.988 | 32.594 | 61.00 |
| FIX_K5，长文 | 232 | 279/1154 | 2.203 | 45.551 | 48.36 |

平均轮次时间=(server.seconds-server.prefill_seconds)/rounds，是总decode阶段摊到轮数的平均值，不是测得单个forward。服务端自报速率另行保存；code K5客户端约82.94，受输出首包/缓冲边界影响，不能拿客户端值掩盖服务端退化。

实际深度已变化：K5约4.99候选/轮，默认代码3.46、长文1.63。故不是变量未生效。FIX_K仅覆盖候选置信度，再经allocator；不提升候选正确率、不保证接受5。多验证的成本超过有效输出增量，本次筛查否决；没有继续浪费重复负载。

## 四、质量与稳定性

quality-summary.json记录相同请求的server token_sha、completion count、finish与输出文本一致。A/B代码与长文各512 tokens，thinking128、工具调用47tokens均一致。FIX_K5四项同样一致；有限样本不是普遍分布证明，也不替代量化quality基准。server hash仅12hex字符，另外检查文本/计数。

原始SSE时间数组保存；SSE不保证一chunk一token，gap/P99仅称SSE-gap，未宣称token-P99或200+滚动峰值。

profiler失败证据：profile-rank1.txt:39开始cuEventSynchronize CUDA_ERROR_INVALID_VALUE，follow_refused CudaFailed。BP数据全部排除，未取得可信prefill阶段拆分。SafeAllocator泄漏消息为退出日志，运行态是否泄漏未证，不据此推断常驻泄漏。profile开关移除、正常服务重载后的两机检查无新CUDA错误。

## 五、修正目标条件【理论推导】

- 本负载prefill1770→3000须总耗时减约41%；16.6K约9.4s需降到约5.54s，约3.8s缺口。未测阶段占比前，不能指定注意力/专家/I/O哪个独自能补齐。
- 代码默认τ=3.34时，200要求T≤16.7ms；当前40.2ms，需降约58%。若T不变则τ需≥8.03。
- 已测K5约46.6ms：即便5候选全部正确、每轮产6tokens，约129tok/s（条件估计，T会随输入/路由改变，不是严格硬件上限）。因此仅调候选置信度不足。
- 稠密/专家/通信/Engram成本需可信实际计数。带宽273GB/s与Tensor理论值不能代替本模型内核的roofline；Mac拐点不可移植。

## 六、下一轮最小实验（尚未执行）

1. 先修复/绕开现有profiling无效CUDA事件：审查live binary与phase event生命周期；profile1或外部Nsight须另设有界实验，不在生产服务留下捕获开关。源码reference目录非Git且未证明和live一致，不可直接编译覆盖。
2. 采集prefill每phase真实时间、传输字节与Engram等待，再单项调整。prefill chunk扩大可能重复读取更少，但会改变舍入、峰值内存、内核捕获覆盖，须先容量账和quality，不盲改。
3. decode以正常run测T、τ：拆草稿/verify/Engram/跨机等待，再选择SIDE/PREFETCH单项负对照或分任务概率校准；默认已经开启，不把默认收益算增量。
4. 高接受率代码held-out集合做峰值筛查，预先定义至少1秒或64tokens以上窗口并用可靠token计数；完整请求速度并列报告，排除重复模板和SSE堆积假峰值。若只靠挑题，不叫系统改进。
5. 200目标长期路线需要草稿质量训练/更低验证成本，不靠FIX_K堆候选；任何新增量化或routing修改单独quality门禁。

## 回滚

远端仅恢复原local.sh备份内容（先核对上述SHA256与目标绝对路径），再通过 `systemctl --user restart dsv41-tensorfold.service`，可回到原250配置。恢复过程中不要改镜像、删除spill/cache、降低reserve或memguard。正常完成应检查两rank存在、profile与FIX_K均缺失、0正确注入、健康接口与质量生成一致。

复现测量：`PYTHONDONTWRITEBYTECODE=1 python3 runs/forge-20261123-interleave/probe.py --arm NEW_UNIQUE_ARM --case prefill16 --rep 2`；同arm原始文件存在则拒绝覆盖。code/prose请求固定用于quality；生产竞争时不拿该运行做单流结论。重新实验须获维护窗口及保证远端有效配置。
