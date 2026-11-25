# E-SELF-10：激活绕过 shared 的真实候选 + 摘部件联合成本判决

日期：2026-11-25　执行：head Spark（192.168.1.118），离线单内核回放
形状：N=2304（全宽）、P=14336、K=5120、R=2048、slots=7、k2=6、活跃组 416/602、活跃程序 7488
计时口径：**CUDA event min-of-min**（ncu duration 只用于取计数器，绝不用于跨臂比时长）
生产状态：实验前后各停/启一次，结束时 `check-boot RESULT: OK`（avail 10 GiB）

---

## 0. 判决表

| # | 候选 / 假说 | 干预 | 结果 | 判决 |
|---|---|---|---|---|
| 1 | **激活绕过 shared 直取 L2**（E-SELF-09e 指认的「真正杠杆」） | MODE 7：删激活 cp.async staging，A 操作数按 `ldmatrix.x4` 语义手工从全局取 4×4 B | **位一致 True / maxdiff 0**，但 **74.294 ms vs 22.719 ms = 3.27× 慢** | **否决（实现正确，性能灾难）** |
| 2 | 「6.06 ms 全在 staging 往返，算术只占 11.2%」（E-SELF-09 §1 原结论） | MODE 6 = 无算术 **且** 无同步，补齐 2×2 完整析因 | 联合摘除 **−4.304 ms**，单独摘除各 −0.73 / −0.84 ⇒ **超加性 2.74×** | **原结论作废，见 §2 更正** |
| 3 | 「per-SM MIO/LSU 队列把有效流水深度钉死在 ~1 级」（我在 09e 后提出的归因） | MODE 6 下扫 S∈{2,3,4,6} | S=2 **+8.5%**、S=3 +11.3%、S=6 +13.3%，S=4 最优 | **否证（深度确实起作用）** |

**一句话结论**：这个内核是一条**已经配平的流水线**——每 stage 的「算术耗时」与「等同步/内存落地耗时」几乎相等（3.461 vs 3.578 ms），二者互相重叠掩盖。所以单独动任何一侧都只有 ~0.8 ms 收益，而两侧一起摘才露出 4.3 ms。**DRAM 62% / Tensor 46% 都不饱和是配平的结构性后果，不是某个单点没调好。**

---

## 1. MODE 7：候选实现与否决机制

### 1.1 实现

`ldmatrix.sync.aligned.m8n8.x4` 配合原 kernels 的地址映射
`lrow = (lane&7) + ((lane>>3)&1)*8`、`lcol = (lane>>4)*8`
产出的寄存器布局为：

| 寄存器 | 行 | k 偏移 |
|---|---|---|
| `a[0]` | `lane/4` | `(lane%4)*2` |
| `a[1]` | `lane/4 + 8` | `(lane%4)*2` |
| `a[2]` | `lane/4` | `(lane%4)*2 + 8` |
| `a[3]` | `lane/4 + 8` | `(lane%4)*2 + 8` |

kernel 内已有 `g8 = lane>>2`、`t = lane&3`，正好对应。于是 MODE 7 删掉激活的 cp.async staging，改为四个 4 B 全局加载直取上述位置。**累加顺序与数值均未改动**，因此正确性闸是「与 MODE 0 位一致」。

### 1.2 正确性闸（`raw/probe-bypass2.json`）

```
基准 max|Z|=412.0444 非有限=0
  MODE 6: 位一致=False maxdiff=412.044  (仅计时,免检)
  MODE 7: 位一致=True  maxdiff=0        (必须位一致)
```

⇒ **寄存器布局推导完全正确**。这是一个语义等价的实现，慢是纯性能问题，不是算错。

### 1.3 性能与机制（同场 A/B，`raw/probe-bypass2.json` + `raw/mode7.csv`）

| 量 | MODE 0/5（shared 路径） | MODE 7（直取 L2） | 倍数 |
|---|---:|---:|---:|
| event min-of-min | 22.719 ms | **74.294 ms** | **3.27× 慢** |
| `global_ld` sectors/request | 17.64 | **7.41** | 0.42× |
| `global_ld` 指令数 | 12,979,008 | **197,032,896** | **15.2×** |
| `ldsm` 指令数 | 76,677,120 | 0 | — |
| L2 读 | 10.184 GiB | 9.387 GiB | 0.92× |

stall 分布（MODE 7）：**long_scoreboard 86.4%**、wait 9.6%、short_scoreboard 3.0%、barrier 0.7%、mio_throttle 0.2%。

**机制**：MMA 的 A fragment 布局天然是「每 lane 4 B、一个 warp 横跨 8 行」。
- 走 shared：全局→shared 是 **16 B/lane 满合并**的 cp.async 批量搬运；读回是 **16 B/lane 无 bank 冲突**的 ldmatrix，短延迟。
- 绕过 shared：散乱布局直接暴露给 L2 ⇒ 每 warp 触 8 行 × 16 B = 8 sectors（实测 7.41 ≈ 8），指令数 15.2×，且**每一条都是压在关键路径上的长延迟访问**（long_scoreboard 86.4%）。

⇒ **shared 往返不是浪费，它是「布局变换器 + 延迟缓冲器」。** E-SELF-09e 里「屋顶内核直取 L2 能跑 429 GB/s，所以 L2 承接得住 8 倍流量」的推理**成立但不可迁移**：屋顶内核是自由布局的顺序流式读，不受 MMA fragment 布局约束。**这是本轮最重要的方法论教训（见 §5）。**

---

## 2. 更正 E-SELF-09 §1 的时间账（重要）

> **⚠️ 口径限制批注（E-SELF-11）：本节的 2×2 不是干净的 2×2。**
> 四格中 MODE 0（算术Y同步Y）与 MODE 5（算术Y同步N）的 `acc` 活跃、**保真**；
> 但 MODE 3（算术N同步Y）与 MODE 6（算术N同步N）的兜底 sink 被死代码消除，
> **连 `asm volatile` 的 ldmatrix 与 `ts[]` 读一起被删**（SASS 实证 LDSM=0、LDS 与 MODE 0 相等）。
> ⇒ **「算术」这个因子捆绑了「shared 读流量」**，超加性 2.74× 不能全部归给「算术 × 同步」的交互。
> 本节数值全部保留，此批注为其口径限制。**仍然保真的部分**：MODE 4/5 只摘同步、acc 活跃，
> 故「同步依赖代价 ≈3.8%」有效；E-SELF-11 的 MODE 9（位一致）进一步证明**栅栏指令本身≈免费**。
> 详见 `runs/forge-20261125-self-decouple/REPORT.md` §1.3。

### 2.1 同场 2×2 完整析因（`raw/factorial.json`，12 臂交替，MODE 0 出现 6 次）

| | **有同步** | **无同步** |
|---|---:|---:|
| **有算术** | MODE 0　**22.7459** | MODE 5　**21.9034** |
| **无算术** | MODE 3　**22.0204** | MODE 6　**18.4423** |

（单位 ms，event min-of-min，全部同一次会话）

主效应：
- 摘算术（同步在）：−0.7255 ms（−3.2%）
- 摘同步（算术在）：−0.8425 ms（−3.7%）
- **两者都摘：−4.3036 ms（−18.9%）**
- 单独效应之和 1.5680 ms，联合 4.3036 ms ⇒ **超加性 2.74×**

条件效应（把交互项摊开）：
- 算术在**无同步**下的成本 = 21.9034 − 18.4423 = **3.4611 ms**
- 同步在**无算术**下的成本 = 22.0204 − 18.4423 = **3.5781 ms**
- 两者相差仅 3.4% ⇒ **配平**
- 互相重叠掉的量 = 3.4611 + 3.5781 − 4.3036 = **2.7356 ms**

### 2.2 同场重标定的纯读屋顶（`raw/roof-final.json`）

`full` 字节集 = 权重 3.428 + 激活 2.928 = **6.356 GiB**（激活 live 行 17056 被 9 个列块各 staging 一遍）
min **15.9018 ms**，达成 **429.2 GB/s**。

### 2.3 更正后的分解

总差距 22.7459 − 15.9018 = **6.8441 ms**：

| 成分 | ms | 占比 |
|---|---:|---:|
| staging 机器（cp.async + shared 往返，无算术无同步） | **2.5405** | **37.1%** |
| **算术 + 同步（联合，互相掩盖）** | **4.3036** | **62.9%** |

> **更正批注（指向 E-SELF-09 REPORT §1）**
> 原文写「差 6.819 ms 中算术仅 0.762（11.2%）、staging 往返 6.057（88.8%）」。
> **该分解作废。** 正确为「算术+同步 4.304（62.9%）、staging 2.541（37.1%）」。
> **错因**：当时只做了「逐个摘部件」，而这两个约束**互相掩盖**——摘掉任一个，另一个立刻完全暴露并继续封顶，于是把联合成本几乎全部误记到了没被摘的那一侧（staging）。
> **纪律**：摘部件法必须先补齐 2×2 完整析因再谈占比；缺一个交互格就会系统性低估联合成本。已写入手册 §7.14。

---

## 3. 否证我自己的「队列钉死深度」假说

E-SELF-09e 之后我提出：S 加深无效、MINB=2 更慢、KB=8 无收益，都指向「per-SM MIO/LSU 队列把有效在飞深度钉死在 ~1 级」。

**可证伪的预测**：若深度真被钉死在 1 级，则在**无算术**的 MODE 6 下，S=2（`wait<0>`，等全部落地，等价无重叠）应与 S=4 一样快。

实测（`raw/m6-s{2,3,4,6}.json`，每臂后紧跟 MODE 0 同场对照）：

| S | wait 深度 | min ms | vs S=4 |
|---|---|---:|---:|
| 2 | `wait<0>` 等全部 | 20.0531 | **+8.5%** |
| 3 | `wait<1>` | 20.5702 | +11.3% |
| **4** | `wait<2>`（默认） | **18.4781** | — |
| 6 | `wait<4>` | 20.9419 | +13.3% |

⇒ **假说否证**。流水深度确实起作用，且默认 S=4 已是最优。

**顺带一个反向交叉验证**：S=3 在**无算术**下比 S=4 慢 11.3%，而在**有算术**下比 S=4 快 1.2–1.3%（E-SELF-09 与本轮附带复现各一次）。**同一参数在算术存在与否时最优方向相反**——这只有配平结构才能解释，进一步支持 §2 的结论。

---

## 4. 对「先把内存吃满，或者把算力吃满」的回答

**做不到，而且是结构性的做不到。**

在一条配平的流水线里，内存侧与计算侧互相封顶，谁也不能跑到前面去：
- 让内存跑更快 → 计算立刻成为唯一限制，DRAM 利用率不会再涨；
- 让计算跑更快 → 内存立刻成为唯一限制，Tensor 利用率不会再涨。

所以 **DRAM ~62%、Tensor ~46%、SM ~50%、L2 ~25% 同时不饱和，是「两侧配平」的必然表现**，不是三个并列的毛病，也不再是「上游某个单点卡住」——**上游没有单点了，两侧本身就是互为止点**。这是沿因果链回溯到底之后的终点。

**要吃满其中一侧，唯一的办法是打破配平**，即改变「每搬运一个字节所对应的算术量」。当前可动的最大一项：**6.356 GiB 里有 2.928 GiB（46%）是激活被 9 个列块各重复 staging 一遍**。减少列块数只压内存侧、不动计算侧，能把配平推向「计算成为唯一限制」。

- 形状约束：`M2_COLS` 必须整除 N=2304 且满足 warp 映射。可选 384（6 块）、576（4 块）、768（3 块）。
- 代价：每 warp 的 n-tile 数从 2 增到 3/4/6，`acc` 寄存器同比增长（当前 REG 133，1 block/SM 下上限 256，有余量）。
- 量级预估（**未实测，仅按字节比例推的待验证假设**）：列块 9→6 使总字节 6.356→5.38 GiB（−15%），staging 侧 2.54 ms 按比例缩到 ~2.15 ms；但配平下总时长由两侧较大者决定，**收益上界受计算侧 3.46 ms 封顶**，不可按字节比例连乘。
- **且 mma2 本身仍不能上生产**：`N%256==0` 硬约束挡住我方 TP2 本地 N=1152，与 v1 非位一致（rel 7.119e-06）。

---

## 5. 边界与本轮自纠

1. **屋顶可迁移性错误（本轮最重要的教训）**：E-SELF-09e 由「屋顶内核直取 L2 达 429 GB/s」推出「激活可以绕过 shared」。**带宽承接得住 ≠ 布局承接得住**。屋顶内核是自由布局顺序读，真实内核受 MMA fragment 布局约束，二者不可迁移。已写入手册 §7.14。
2. **dispatch 顺序 bug（跑前拦下）**：`TF_PROBE_M2_S` 的分支排在 MODE 链之前，`S=3 MODE=6` 会静默命中**完整算术的 S=3 臂**。已给 S/MINB/KB 分支全部加 `pm == 0` 门控，并用 mangled 名逐臂断言 4 个 MODE 6 实例（`Li2E/Li3E/Li0E/Li6E` + `Li1ELi6ELi4E`）各存在 1 个。**若未拦下，整个 §3 的判决会是假的。**
3. **驱动缺正确性闸**：`probe-m2-parts.py` 原先**没有**位一致闸（闸在 sweep 驱动里），我一度以为 MODE 6/7 已过闸。已补：MODE 7 强制断言位一致，MODE 1–6 明确标注「仅计时、免检」。
4. **cuobjdump 名字是 mangled 不是 demangled**：先前按 `<2, 6, 6, 64, ...>` 的 demangled 形式 grep 全部落空。正确形式为 `_ZN8tf_exl3x19grouped_mma2_kernelILi0ELi6ELi6ELi64ELi{S}ELi{MINB}ELi{MODE}ELi{KB}E...`。
5. **MODE 1–6 是故意竞态/无算术的只计时探针，输出不对，绝不可上线**；MODE 7 语义正确但慢 3.27×，同样不上线。**本轮没有任何改动进入生产**，生产仍是 `base-alias` 基线。
6. **归档纪律**：E-SELF-09 的 `src/experts_grouped.sweep.cuh` 已 `git checkout` 还原为提交态；本轮演进版另存 `raw/experts_grouped.sweep.cuh`（sha256 `bdc4d4a0bf84192bd77764c1a1dc2693e301edd25a150dbe9f71a8ab85e2c1c9`）。
7. **未复验**：全部为空载单流离线回放；生产多路（head 最小空闲 4.8 GiB）下未复验。

---

## 6. 复现

```bash
# Spark head，停服后
docker run -d --name exl3-probe --gpus all --entrypoint bash \
  -v /home/ztxjzh/ptxwork:/home/ztxjzh/ptxwork -e HOME=/home/ztxjzh \
  --cap-add SYS_ADMIN tensorfold-dsv41:exl3-tp2-8348338a8d90 -c "sleep 14400"
cp raw/experts_grouped.sweep.cuh ~/ptxwork/exl3src-sweep/experts_grouped.cuh
cp raw/probe-m2-parts.py         ~/ptxwork/probe-m2-parts.py

docker exec exl3-probe bash -c 'cd ~/ptxwork && python3 probe-build-sweep.py'   # ~75 s

# 2×2 完整析因 + 候选（带正确性闸）
docker exec exl3-probe bash -c 'cd ~/ptxwork && \
  python3 probe-m2-parts.py --modes 0,3,0,5,0,6,0,3,0,5,0,6 --reps 6 --no-check && \
  python3 probe-m2-parts.py --modes 0,6,0,7,0,6,0,7 --reps 8'

# 深度判决
for S in 2 3 4 6; do docker exec exl3-probe bash -c \
  "cd ~/ptxwork && python3 probe-m2-parts.py --mode 6 --s $S --reps 8 --no-check"; done

# 同场屋顶
docker exec exl3-probe bash -c 'cd ~/ptxwork && python3 probe-pattern-roof.py --only full --reps 8'
```

收尾：`docker rm -f exl3-probe && systemctl --user start dsv41-tensorfold.service`，
等 `/health` 200 且 avail 落到 10 GiB 后在项目目录跑 `bash tools/check-boot.sh`。

## 7. 原始读数

`raw/`：`factorial.json`（2×2 十二臂）、`roof-final.json`（同场屋顶）、
`probe-bypass.json` / `probe-bypass2.json`（MODE 6/7，后者含正确性闸）、
`m6-s{2,3,4,6}.json`（深度扫描）、`mode7.csv`（MODE 7 计数器）、
`build-sweep6.log` / `build-sweep7.log`、
`experts_grouped.sweep.cuh`（演进版探针源码）、`probe-m2-parts.py`（带闸驱动）。