# Round4：实际专家指令与资源事实

本轮只读，不重启生产。读取当前运行镜像挂载kit/cubins/experts_cb.cubin，用已有cuobjdump解析实际SASS，SHA256 b9a6fa174240d7f1376389989b3124950786eb0503c834a1c94a8e16882fc85f。从此前实际CUPTI mangled name匹配gate/up与down两个模板，不靠reference模型推断。

## 已核实

- gate/up与down均出现HMMA.16816.F32（FP16输入、FP32 accumulator tensor instruction），不是原生NVFP4 block-scaled MMA。
- 两函数有LOP3、SHF、PRMT、SHFL、IDP/HFMA2等大量解码/索引/变换指令；静态代码数量不等于动态执行频率，不能直接报解码百分比。
- 真实trace资源：gateup106 registers/thread，256 threads/block，44288 bytes static shared；down255 registers/thread，128 threads/block，32896 bytes static shared。高register/shmem是occupancy限制候选，实际resident blocks与warp stalls未测。
- trace localMemoryPerThread=0；localMemoryTotal字段75497472不能直接解释为每次spill流量/带宽成本，未据此宣称spill。
- gateup120calls平均10.4627ms(min2.174max16.019)，down120calls平均5.2086ms(min.802max8.021)。各层/尾块形状不同且首请求lazyloading，不能把min/max归因为随机波动或算力变化。

证据expert-sass-selected.json、expert-resources.json。

## 物理意义

当前2.9bit压缩减存储，但实际专家Tensor计算是FP16，加上解码/变换/小M分组。不能用2PFLOPS sparseFP4分母计算利用率；格式换NVFP4有数值/容量问题，不能当无损开关。最值得证伪：解码instruction供给vsDRAM/L2stallvs寄存器资源限制。需匹配真实形状、路由的重放计数器或稳态trace，不能仅靠SASS数量做速度预测。

下一轮先在同进程形状warm后不同noncecached0冷prompt验证稳定headkernel成本，再量动态stall/occupancy。worker工具gate已通过但真实follower采集收尾缺活动尚未解决；可单独traceworker并明确flush顺序，避免把缺数据当慢机器。生产保持原配置健康。
