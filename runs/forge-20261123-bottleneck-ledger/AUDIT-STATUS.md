# 本轮证据审计与待执行项

2026-11-23，goal round6/6。任务未完成根因识别，不标完成。

## 已验证

- 时间/算术账REPORT与对应JSON闭合；Engram关键等待边界、core成本与Amdahl反事实可复算。
- live machine code证明串行两层read、整线程join、每行两次alignedrequest规则；不证明动态gap因果。
- 实际Engram文件geometry只修weightalignment的uniform条件字节收益3.03%；不能把32×请求放大当收益。
- 本轮分析器修正：wait_binding unknown/current_layer时coupling收益输出null，不按whole_reader虚构数值；时钟未明确确认unverified。15fixturetestsPASS，非生产证据。
- 生产原镜像Up58min，无bpftrace/nsys进程；本阶段未创建实验GPU负载/容器/探针，未重启、未提权、未下载。只持久研究材料，无临时计算cache(-B)。

## 必须补的证据

1. 动态j0ready/j1ready、实际wholejoin进入退出与GPUdependency需求时刻；此前4大gap不能继续静态推导唯一原因。
2. 实际专家count/validtile、DRAM/L2 bytes与执行管线测量，区分算力/解码/访存。
3. 两rank消息就绪时标与字节，分离网络与peerwait。

## 最小下一执行建议

优先无重启probe门禁+至多30秒观测：使用现成bpftrace和已经核实binarysymbol/fileoffset，过滤目标PID；先在独立scope验证权限和退出清理，不改系统perf策略；只低事件率ready/join，不逐I/O打印。附加探针会扰动生产，需要明确界限和基线对照，不称zeroimpact。

当前head sudo-n不可用，非特权perf/BPF/tracefs限制存在；没有可直接使用的安全凭据输入接口，不能将已给密码写入命令/脚本或记录。可让用户在自己的终端完成必要的一次权限门禁，或明确批准受控采样入口；技术授权不等于自动放开>60秒窗口。

替代方案仍是诊断artifact准备完后的10–15分钟受控重启采样，时间窗口尚未明确确认。长构建另估时，不能无界后台执行。

本轮不mark blocked以代替研究判决，也不mark complete；目标active，等待用户选择动态验证方式。所有当前结论仍标理论/实测/假设边界。
