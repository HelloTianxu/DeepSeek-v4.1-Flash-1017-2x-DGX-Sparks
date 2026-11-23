# 采样准备进展：生产来源与离线判决器

2026-11-23。未重启、未跑模型或GPU实验、未下载。goal round2。

## 来源核对

当前容器running image sha256:299302f83fd3134d52b8da4b63db0b0fe90a5d449b52f894c07f44f2b9a9ce2b。image标签tf.ref=fbbc97132cd004237d9887b8d3185896efaedc5c、tf.patches=8348338a8d90、tf.inputs=59402086381a42c74d53053443224eec2790b4c5。

现有两个remote源码root的git对象均未找到这个固定commit。Dockerfile明确构建后删除/tensorfold源码；reference/provenance.json的git为空、git_error不是仓库。故当前参考实现身份不足，不能把其whole-reader join作为live已证事实。生产trace中的Engram入口gap仍是真实证据；请求31.03×为参考算法+实际表结构的条件账，O_DIRECT描述符支持读取模式但不证明实际submit规则。

无需因未找到源码而下载/重建整个模型。后续可检查本地BuildKit/cache中的构建来源或live二进制符号/调用栈；找到匹配源码之前，插桩设计只按事件语义，不能把另一版源码替换进生产。

## 已实现离线判决器

analyze_engram_events.py：读JSON，显式标synthetic或measured，不允许省略时钟确认；检查generation、chronology、wait_binding。分别计算主机join、当前供给晚、release envelope及条件coupling；GPU dependency需求时刻与host join_enter分开。不把两rank等待相加，不报告total_gain。

复现：`PYTHONDONTWRITEBYTECODE=1 python3 -B test_analyze_engram_events.py`。12条已知答案测试全部通过：跨层耦合、当前数据晚、host join完全被GPU隐藏、非whole绑定、字段缺失、stale generation、非法顺序、join早退、时钟未校准、需求时刻误差、证据类型必填、跨rank不加和。

测试只验证代码与数学，不证生产根因。源文件持久保留，无临时cache输出（-B）。真实事件尚未采到。coupling计算是固定其他成本的反事实，不能作为实测可回收时间；分析器输出明确限制。

## 下一动作

优先live来源/调用边界匹配，之后插桩准确采actualGPU need、layer ready与reader退出。采样预算/重启窗口仍未得到明确确认。目标active，未宣告已完成根因识别。
