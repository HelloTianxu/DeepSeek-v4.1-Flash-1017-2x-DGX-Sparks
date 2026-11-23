# 生产二进制机制核对：Engram证据链提升

2026-11-23，goal round3。只分析当前生产已有二进制，不运行模型、重启、重建或下载。

## 实际artifact

/opt/tensorfold/native/bin/tensorfold-native，27501920bytes，ELF64littleaarch64，保留.symtab及DWARF.debug_info/debug_line。当前镜像299302f83...，tf.ref=fbbc97132cd004237d9887b8d3185896efaedc5c，补丁8348338a8d90。先前reference无Git来源的问题未通过源码完整重建解决，但现在可针对重要机制用live binary独立核实。

## 直接核实

1. live-fillChunk-disasm.txt地址0x13fb500调用pthread_join；addr2line嵌套栈对应model.fillChunk:1204，恰为参考ahead.thread.join。另0x13fb568对应engram_lead.before的独立层join:1209，不应混为一种路径。
2. 实际预读entry Thread.PosixThreadImpl.spawn__func_3132.Instance.entryFn，0x1403e68把x5(j)设为0，0x1403e6c调用engramAhead1；成功后0x1403e90把j设为1，0x1403ea4再次调用；随后free、return。DWARF这两个调用均对应prompt.engramReadAheadAll:1518，建立实际串行双层reader机制。
3. 实际fillChunk启动上述thread的pthread_create地址0x13fe980，对应model:1227（下一chunk读取All）；0x13fc544对应model:1219当前chunk下一Engram层单读，不应混淆。
4. 实际engramAhead1调用0x140409c -> prompt.aioGather，随后0x1404124 -> engram.decodeRows，再0x1404134发布ready byte，对应prompt:1538。ready在该层读取解码完成后、thread可能继续下一层时发布。
5. 实际Aio.start包含内联submitLocked：0x13d6490把ids.len乘2；0x13d6944以j>>1选行，0x13d6948按j&1选weight/scale；0x13d695c/6960计算4KiB skew/base；0x13d69a0加4095、0x13d69c8对齐nbytes，0x13d69d4存nbytes/offset。该loop对应submitLocked:176–190。因此每行两次独立对齐请求规则有live机器码支持，不再仅reference假设。

## 判决界限

【工程实测：artifact静态核对】串行双层预读、整线程join及每行两次4KiB对齐请求存在于生产二进制，与reference关键行号一致。

【理论推导】结合真实表geometry与TP2列数，2048token/rank/layer24576rows→49152requests最低201.33MB对有效6.49MB，31.03×请求放大仍成立；真实每请求是否走此分支/多少8KiB crossing/完成字节与重复块需动态记录，不能直接报实际drive I/O或31×收益。

【研究假设】若第1层已ready、主机却等待串行线程第2层结束，存在跨层关键路径耦合。二进制证明机制不证明每个gap由其导致；真实当前layer ready与threadexit尚未采。

下一最小采样位置现在可绑定live符号/地址/源行，不必再泛找源码。优先reader j0/j1完成、入口join_enter/exit、GPUdependencytime与generation；源完整身份仍需构建匹配，插桩不能凭相似源码替代生产。可研究uprobes/符号级非侵入采样但附加开销/权限先门禁，未执行。

## 原始证据与复现

live-symbols.txt、live-fillChunk-disasm.txt、live-engramAhead1-disasm.txt、live-aio-start-disasm.txt、live-reader3132-disasm.txt、live-callsite-lines.txt、live-prefetch-lines.txt。复现工具existingcontainer readelf -S、nm -C、objdump -d --disassemble=<symbol>、addr2line -f -i -e <binary> <address>。所有命令有8–22s远端timeout，单工具总<60s。额外两entry3135/3144作为对照保留，其中3144不是All，不能按编号猜功能。

研究目标继续active：时间、算术、位置与机制候选已有证据，但主内核动态瓶颈与Engram因果贡献尚未闭合；没有新性能成绩或改进上线。
