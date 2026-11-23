# Mac oMLX与Spark Engram证据对照

2026-11-23。仅复用本地原始JSON、探针源码与当前storage.py，不运行Mac模型、不改服务。

## Mac原始记录复核

E-ENGRAM-01 CHUNK8192，16K输入：plain median11.8012s，instrumented median11.806s；embedding.__call__ host blocking median.0094s，read_rows host wall median.1202s，prefetched slot present4/4。

.0094/11.8012=.07965%，固定其他成本的反事实11.8012/(11.8012−.0094)=1.000797。因此本实验中Engram调用的host阻塞非常小，优先不做高风险驻留改造有依据。该上界依赖主机阻塞/潜在额外graph enqueue效应不改变其他时间，非GPU实测idle判决。

原始计数rows_requested539760、rows_unique446555、bytes_selected71448800=446555*160。786432为16384*24*2的名义原始行数，tuple dedup后539760，再row unique446555。不能说539760*160=71.4MB，实际为86.36MB。

bytes_selected通过selected_bytes(unique rows)计算，代表有效选中packed weight/scale/bias字节，不是os.pread字节、SSD物理触碰或缓存缺页字节。_read_row_runs会读取间隙，最多64KiB gap合并、最多256rows/run；预读与文件cache可进一步改变实际介质流量。因此“容量/71.4MB=1720×”可描述容量相对有效选中量，不能描述实际磁盘读放大。

pending存在算prefetch_hit，实际__call__还比较requested和host，不同则fallbackread；本次小blocking支持有效预取，但slot-present不是单独充分的匹配命中证明。

## 测量方法边界

probe blocking计时包围orig_call，不强制GPU synchronize：它是host future.result/read+lazy decode graph构建时间。主机不继续前向不意味着GPU必定空闲，GPU可执行已排队工作。用host阻塞作为额外延迟预算比把它称GPU实测stall准确；也不可仅io_wall−blocking就称GPU隐藏92.2%。尤其blocking包含graph构建，io_wall只read，两口径不是严格同项集合。

部分instrument外部np.asarray/np.unique/bump不在blocking计时内，但进入instrumented总T；CHUNK8192plain/instrumented median差4.8ms，在样本散布内；SINGLE16384instrumented13s显著较plain12s，不能宣称整个探针完全无扰动。没有GPU时间线证明三数量级差。

## 实现差异

Mac：oQ4e affine4bit rows160bytes（含scale/bias），48并发pread、row np.unique、tuple dedup、64KiB gap-run merging、跨chunkcarry；prefetch管理本身单worker，但底层readruns48worker。

Spark：EXL3专家checkpoint+FP8Engram rows264bytes，每rank12cols，2048chunk、成对4KiBalignedO_DIRECT、单reader串行双layer，实际24io_submit×2048requests=49152reads/layer/rank，blk_mq tagwait动态实测；无需假定“完全没有并发”，LinuxAIO本身并发但提交thread/tag局部背压存在。两边数学模型名称相同不等于packed格式/工作量/配置/软件路径相同。

Spark已有headidle约.664s Engram入口（旧同GPUtrace），没有同请求hostready映射或worker同步，不能将.664/.0094=70.6×当公平性能比，也不能以Mac.0094反证Sparkwait不存在。Mac结果提供可复用机制方向，不是Spark收益乘数。

## 改进研究方向

先复现Mac-style统计口径到Spark：真实IDs、tuple/row唯一率、完整pread/IOCBrequested bytes、完成bytes、缓存状态、readruns与batch；再测同请求GPU关键路径、两rank就绪与observer配对。

优先因果实验：保持同输出与ring，仅改提交burst，区分tagwait减少是否使totalread减少；随后单独测dedup/runmerge或层级reader就绪。不要因Mac已有48threads直接把Spark改48reader，也不要把有损量化格式移植当无损优化。

最终判决：Mac这次测得hostEngram阻塞小，支持当地无需优先优化它；Spark块层/tag供给机制已有实证，但可回收GPU时间仍需确认。双方不矛盾，测量表述必须更严谨。

原证据本地熔炉项目experiments/prefill/forge-engram-20261123-r2/engram.json与scripts/probes/forge_engram_probe.py，oMLX storage.py；本轮所有数字仅从这些读回计算，没有新实测。
