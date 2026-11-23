# 计数器门禁与上传等待负对照

2026-11-23，用户要求继续，一切独立微测试不重启生产，不改系统权限。

## ncu门禁

4MiB独立kernel，原image --rm networknone，ncu --set basic --launch-count1返回ERR_NVGPUCTRPERM。未开启系统性能计数器权限，未对生产进行multi-passreplay。counter-gate.txt为负证据，不能因此宣称Tensor/DRAM利用率。

## pageable/pinned机制负对照

先在同一stream提交7000000 GPU clock循环小kernel，然后上传4KiB；30次meanhostcall pageable0.002284ms、pinned0.001831ms。单变量hostallocationtype。源/编译/运行在独立--rm容器，不污染生产，无大分配。pageable-copy-gate-v2.txt。

第一次编译失败：CUDA13 cudaDeviceProp不再有clockRate，保留pageable-copy-gate.txt；第二次用固定GPUclockcycles修正，不是可定时3ms证明。没有试图从clockcycles推性能。

结果不复现模型HtoD累计2.6s，故“所有pageable async upload天然等待先前kernel”不是本机普遍成立的机制。不能据此直接上线pinned改造。模型upload不同大小/stream依赖/context状态，需真实调用受控重放。

当前已确认：warm形状后libraryload0，但gateup1.28s down.63s持续；实际HMMA.F32累加路径+整数解码；寄存器候选无计数器实测；HtoD物理copy短hostcall长未最终归因。下一步隔离真实expert数据/路由重放与大/小upload匹配，不做无证据optimizations。
