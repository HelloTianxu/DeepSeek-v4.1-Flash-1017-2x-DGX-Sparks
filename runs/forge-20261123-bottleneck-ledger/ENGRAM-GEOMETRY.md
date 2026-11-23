# Engram对齐与去重的收益边界

2026-11-23，goal round5。实际两表safetensors头已读取，未读tensor负载、未生成请求、未重启。

## 精确几何

层1weight起点664(mod4096)，layer14起点672；行宽256。两者均为row_id mod16=13时跨4KiB边界，需要8KiB请求，其余4KiB。scale行宽8，实际起点2712/3232 mod4096均8对齐，无跨块。

【理论推导，实际文件geometry】每层每rank2048token共24576rows，最低4KiB-per-weight+scale request201326592bytes。若row_id mod16均匀分布，1/16weight跨块：weight平均4352、scale4096，总8448bytes/row，207618048bytes/层/rank，相对264有效bytes为32×。均匀residue未实测，故不是32×实际流量。先前31.03×仍是最低请求放大而非均值。

## 修对齐只能收回小部分

假定row_id residue均匀，仅把weight起点256对齐且其他行为不变，减少8KiB跨界附加6.291456MB/层/rank，即请求字节3.03%。这是字节差不是时间收益。不得说“消除32×放大”；小行随机4KiB读取仍保留至少31.03×。

O_DIRECT块粒度是当前读协议/OS界面限制，不代表SSD必须为每重复请求读盘。去重只有在实际hash/block重复时才有效；每块可装16weight行或512scale行并不表示当前随机采样就访问了这些相邻行。必须测Ublocks/Qrequests及repeat分布，不能拿512当scale读取加速倍数。

## 条件性随机模型用于检查预期，不做判决

若m个行在N行全空间独立均匀，块数B≈N*row_bytes/4096，期望unique blocks≈B*(1-exp(-m/B))，但实际Engram分column hash与重复prompt并不满足该模型。以N约384M,m24576，scale B约750k，m/B约0.033，期望去重比例仅约1.6%；weight B约24M，比例约0.05%（未计跨块）。这解释为什么“行小/一块装很多行”不保证块去重大收益。真实重复文本可能显著改变结果，需要实际ids。

## 最小验证

从真实reader在计时外捕获匿名row IDs或提交跨度，按(file identity,aligned block)离线统计：总requests、requested bytes、unique blocks、同层/跨层/跨chunk重复、row_id mod16比例。包含8KiB跨度的两个block，不能把read起点当完整跨度。原输入文本不记录；row IDs属工作负载信息，按用户数据隔离保存。

先计算请求可减少量，再做读取/scatter的逐byte等价实验。若duplicate低就否决dedup优先级，转layer ready同步或提交/回收开销。若仅对齐3%字节收益，不为其重写200GB表。

复现：python3 -B calculate_engram_blocks.py（只读JSON、计算周期，产出engram-block-geometry.json）。实际offsets见engram-absolute-offsets.json。所有均匀与去重数字标记假设，无优化成绩。
