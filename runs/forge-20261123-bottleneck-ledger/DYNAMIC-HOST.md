# 首次动态Engram事件：主机耦合确认，GPU收益未测

2026-11-23约16:30–16:40。用户明确授权继续且此前批准受控诊断。此次采用更小扰动无重启20s函数探针，未改模型/路由/生产启动配置。

## 过程与负结果

privileged临时容器nsenter hostmount运行现有bpftrace BEGIN门禁成功；不改系统权限参数。第一script嵌套引号SyntaxError，无probe挂载，请求HOSTREADY为正常无trace基线。v1数字fileoffset被bpftrace按VMA解析失败；v2改VMA后本机bpftrace无instructionoffset支持拒绝；均cleanupabsent。v3仅函数边界挂载成功但手动request窗口未同步，空数据不代表没有行为。v4收到READY立即由脚本触发请求，取得事件。没有绕过instructionboundary检查；改为受支持函数入口/返回与libc pthread_join探针。

原始engram-host-trace.txt/-v2/-v3/-v4保留负/正证据，trace_engram_host.py为最终同步版本。参数ABI实测j0/j1,start0/2048/4096/6144/8192,n2048/396与真实request一致。探针exit记录retval0（成功），不是指令级ready timestamp；实际binary证明ready发布在函数成功返回前。pthread_join包含decode线程joins，按main_tid独立筛选，不能把所有join加到请求时长。

## 动态结果

HOSTSYNC4请求N8588cached0prefill4.950441s，首tokenhash528b0f3da89f与早基线相同。不是质量/等分布完整验证；单请求未做observer对照。

|chunk|rows|j0毫秒|j1毫秒|main join毫秒|j0启动至join进入毫秒|同reader|
|---:|---:|---:|---:|---:|---:|---|
|0|2048|242.61|235.21|242.20与233.00|0.52|否，分层两个thread|
|2048|2048|233.80|234.83|467.04|1.67|是|
|4096|2048|243.41|240.66|482.94|1.34|是|
|6144|2048|229.93|224.81|453.71|1.21|是|
|8192|396|51.22|48.35|97.76|2.09|是|

关键更新：j0成功返回晚于join进入约232/242/229ms，所以“进入join时第1层已ready”严格条件被本请求否定。但j0完成后还等待j1约235/241/225ms，whole-reader同步耦合真实存在。不能由此说GPU多等0.70s；主机线程提前入join，GPU还可能执行前chunk，所以必须实际GPU需求时标。

## 下一判别

1. 同源GPU边界与host时钟对齐，看j0ready是否在真正GPU需要第1层之前；若在之前而j1之后，则可回收等待明确。
2. 分解约230ms perlayer为hash/AIOread/decode；当前function span包括全部，不能归SSD。
3. 首chunk与后续配对reader分开，首chunk无法用跨chunk预取解释。最后tail同样可能被GPU覆盖，不把97ms当critical成本。
4. 下一观测必须paired重复并测observer开销，确认两rank同步。现有数据已排除纯静态推测，尚未最终瓶颈判决。

## 清理

每门禁/trace用--rm唯一容器名，finally查容器存在则精确rm-f该名并复核。每次输出CLEANUP_ABSENT True。未重启原服务，没有driver/module/perf策略改动；研究数据持久保存。

注意整个准备期间约10分钟，不等于10分钟连续probe；实际单次probe20s，各command<60s。四个background jobs已job_output收集；失败非隐瞒，无旧任务宣称运行。
