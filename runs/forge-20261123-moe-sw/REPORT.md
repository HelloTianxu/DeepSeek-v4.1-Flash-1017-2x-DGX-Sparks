# MoE细拆第一份有效kernel时间线

2026-11-23，目标prefill3500，先诊断不改算法。用户批准10–15分钟最多2次两机重启；已正常回退。原start.sh/local.sh不改，临时启动副本+dropin包装nsys cuda-sw,nvtx，CPU采样关闭，graph粒度不做node。外层shell转发TERM并等待nsys完成，取消固定duration；正常停止保存原报告。此前生命周期故障本轮未重现。

## 有效范围

SWTRACE冷技术文档8531tokens、cached0、max_tokens1，完整返回hash528b0f3da89f。服务端prefill5.640342s，tracehead选取startup后最大23sidle gap的后续kernel窗口5.632372s，与请求吻合。仅head产生有效CUPTI活动：6104kernel、71323API、50944memcpy(包括加载)；worker报告仅NVTX，无CUDAkernel表，原因未知。不能声称双侧对照完成。

## head请求内kernel累计时间【工程实测】

| kernel/类 | 累计ms | 调用 |
|---|---:|---:|
| EXL3 grouped_mma gate/up |1255.529|120|
| EXL3 grouped_rows down |625.036|120|
| NCCL AllGather_RING_LL |530.227|251|
| dense `_gemm` |457.030|795|
| `_hc_mix_pf` |280.474|226|
| `_sparse_attn_part` |258.402|120|
| 专家rot_in |146.478|120|
| 专家down_combine |130.505|120|
| dense unpack |95.915|780|
| gateup_epilogue |64.584|120|

代码关联参考：exl3_experts.prompt / gateUp / down对应上述manglename与实际参数；不是原生NVFP4路径。最终是否bound于DRAM、EXL3decode指令或MMA利用率仍须计数器/隔离重放，不能凭name推算。

最小主战场细拆：仅gateup+down=1880.565ms，专家rot/combine/epilogue额外341.567ms；已确认专家计算kernel显著，不再只有粗MoE桶。以上kernel内也可包含等待数据/内存，不是纯Tensor乘法时长。NCCL530ms含等对端，不能等同线速copy或全部MoE内部exchange。

所有selected kernel累计4309.188ms，union4292.885ms，窗口5632.372ms，no-kernel1339.487ms。空隙包含memcpy/host/unrecorded等待，不一定可全部消除；累计不直接相加断言关键路径。只看head不能量两rankimbalance。

## API等待与冷态污染

请求窗口内cuMemcpyHtoDAsync505次累计2804ms(max623ms)，cuLibraryLoadData26次累计952ms，cuKernelSetAttribute95次400ms，cuStreamSynchronize1次143ms。API与GPU工作可重叠，不能在kernel总时间上直接加；HtoD调用很慢不证明传输很慢，pageablehostmemory、隐式同步或库加载可能参与。

重要：这次是服务重启后的首个对应形状prefill，包含lazy load/JIT；cold prefix不等于steady warmkernels。不能将952ms libraryload都称日常稳态成本。下一最便宜实验是在同traceprocess内先形状预热、再另一nonce同tokenlengthcoldprefill，比较库加载与kernelcost；不得用热prefixcache造加速。

## 未完成项

worker cuda-sw仍未产生活动表，diagnostics记录NoCUDAevents。gate小程序只head验证过；必须worker独立gate以区分节点工具与TensorFoldfollower特异失败。没有证据归因为网络/driver差异。要在下一授权窗口复测对照，或用prefill-onlyevents专门对workerMoE子段测量。

## 回退

原ExecStart ./start.sh已恢复，dropin90-forge-sw.conf核对内容与绝对路径后删除，临时start副本及wrap文件SHA核对删除，空dropindir删除。两rankrunning原image8348338a8d90，无nsys进程、近期log无CUDAerrors。SWRESTORE code512hash1428664efda9匹配原始基线。原model/cache/clocks/guards保持，无新优化落地。

原始证据：head-kernels.json、head-kernel-summary.json、head-api-summary.json、head-coverage.json、head-trace-schema.json、worker-diagnostics.json、cleanup.json；remote各~/.cache/forge-moe-sw-20261123/trace.nsys-rep和sqlite为保留研究证据，不在modelcache。导出使用已有原image --rm networknone，不下载。

结论：主瓶颈拆到EXL3 gate/up和down真实kernel；还伴随NCCL及host空窗/lazyloading。根因级computevsbandwidth/对端等待尚未证，3500未达，下一步必须测稳态窗口与worker对照，不盲调FP4/频率/深度。
