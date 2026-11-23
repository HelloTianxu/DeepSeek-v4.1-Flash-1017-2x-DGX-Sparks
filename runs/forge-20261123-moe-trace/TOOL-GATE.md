# CUDA追踪工具门禁通过

2026-11-23 14:52，未重启生产、未改系统权限。复用生产同一镜像8348338a8d90，在独立--rm、network none容器中执行4MiB CUDA测试。第一项ctypes Driver API 200memset，第二项nvcc sm_121编译100次真实kernel；各命令<55s，无SDK/网络下载。

结果：`--trace=cuda-sw --cuda-graph-trace=graph --sample=none --cpuctxsw=none`导出sqlite有真实活动表：Driver测试205runtime/200memset；kernel测试207runtime/100kernel/1memset/1synchronization。见cuda-sw-gate.txt、cuda-sw-kernel-gate.txt。

这证明该软件追踪路径在本机、同镜像、同Driver环境可用。尚不能证明真实TensorFold/两机图追踪完全支持；prefill使用普通launch，可以不要求graph-node追踪。上轮默认hardware路径无CUDA数据原因尚未完全证实，但不必再重复同失败路线。

微型容器退出时自身/tmp源码、二进制、nsys临时报告随--rm清理；保留本地stdout证据。docker ps只剩原生产容器，health ok，没有nsys/forge_gate进程。原配置不变。

下一步真实负载必须先修生命周期：不能把有限duration的nsys直接作为容器PID1；由外层包装监督应用，或在用户授权窗口中将capture与正常进程退出绑定。自动健康就绪立即触发唯一coldprefill max1，不能人工检查耗尽捕获窗口；失败回退原unit并验证两rank。若需要超过1分钟，先取得新的时间/重启授权。
