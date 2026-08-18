# Looped Transformer 不只是重复计算：从 LoopFormer 到 Elastic-Depth Post-Training


Looped Transformer 最容易被误解成一句话：把同一组 Transformer block 多跑几遍，用时间换参数。

这句话没有错，但它跳过了真正困难的部分。共享参数只回答了“循环什么”，没有回答另外三个问题：模型如何知道自己处在第几次循环？短预算为什么仍然能给出有用结果？增加循环时，hidden state 为什么不会漂移、坍缩或者干脆变差？

把近期工作放到同一张图里看，我更关心的研究命题不是再从头训练一个固定循环次数的模型，而是：

> **Elastic-Depth Post-Training：能否把已有的固定深度或固定循环模型，改造成计算预算可调、迭代过程稳定、并能按样本自动停止的模型？**

LoopFormer 给出了全局预算和 trajectory consistency；LoopUS 展示了如何改造 pretrained LLM，并用 selective update 与 confidence exit 控制迭代；Relaxed Recursive Transformers 则提供了“共享主参数、保留深度角色”的折中。三条路线尚未被组合验证，但它们拼出了一个比“多循环几次”更完整的研究问题。

本文信息截止到 **2026-08-18**。文中的数值若无特别说明，均为相应论文在其模型、数据和硬件设置下的报告，不能直接外推到其他模型规模或真实部署吞吐。

<!--more-->

## 1. 30 秒结论

先给出五个判断。

1. **参数共享不等于弹性深度。** 固定用同一组参数执行两次，与用户可以在推理时自由选择一次、两次或四次，是两个不同问题。
2. **LoopFormer 的关键不是 loop，而是 trajectory。** 它把多次循环视为从时间 0 到 1 的状态演化，并让每一步显式接收当前位置与步长。
3. **全局预算和自适应预算需要分开。** LoopFormer 为整个 sequence 选择预算；MoR 在 token 级路由；LoopUS 用 confidence head 做样本级 early exit。三者的系统复杂度完全不同。
4. **已有 checkpoint 可以成为起点。** Retrofitted Recurrence、Relaxed Recursive Transformers 和 LoopUS 都说明“从 pretrained model 出发”是一条真实存在的路线，但改造方式、训练成本和能力保留仍需逐模型验证。
5. **最值得做的不是直接把 Qwen 的层硬循环。** 更可检验的方案是同时处理初始化、深度条件、短长轨迹一致性、选择性更新和停止策略，并以“短路径可用、长路径单调改进、原能力不遗忘”为联合验收标准。

这不意味着 latent reasoning 已经可以替代显式 Chain-of-Thought，也不意味着参数更少必然带来更低延迟。循环减少的是存储的独立参数；实际执行的 FLOPs、KV cache、memory access 与批处理效率仍由具体实现决定。

## 2. 读完后应能回答

- Looped Transformer 与普通加深、early exit、Mixture-of-Depths 有什么区别？
- LoopFormer 如何让同一模型支持不同循环预算？
- 为什么循环次数增加后，模型可能变差而不是变好？
- 哪些工作适合从头预训练，哪些更适合已有 checkpoint 的后训练改造？
- 如果从 Nanbeige4.2 或 Qwen 出发，最小可证伪实验应该怎么设计？

## 3. 先统一几个概念

设一个模型包含 $k$ 个物理 block，这组 block 被执行 $R$ 轮。

| 概念 | 含义 | 容易混淆的边界 |
| --- | --- | --- |
| physical depth | 实际存储的独立 block 数量 (k) | 决定参数存储，但不等于执行计算量 |
| unrolled / effective depth | 展开后的执行深度，近似为 (kR) | 循环越多，FLOPs 通常也越多 |
| global budget | 整个输入共享同一循环次数 | 实现简单，但不能给困难 token 单独加计算 |
| example-level exit | 不同样本在不同轮停止 | batching 时会出现样本完成时间不齐 |
| token-level depth | 不同 token 进入不同递归深度 | 需要动态 token 集合、选择性 cache 和更复杂调度 |
| elastic depth | 同一模型在多个执行深度下都可用 | 不只是“测试时可以强行多跑几次” |
| latent reasoning | 在 hidden state 中增加计算 | 不保证形成可读、可监督的推理过程 |

一个最简单的 Looped Transformer 可以写成：

$$
h_{r+1}=F_\theta(h_r),\qquad r=0,1,\ldots,R-1.
$$

这里每轮都使用同一个 $F_\theta$。问题也正出在这里：如果函数看不到轮次，模型就必须用同一套行为同时承担浅层表征抽取、中层组合和深层决策。训练只见过固定 $R$ 时，测试时改成另一个 $R$ 也属于分布外执行。

## 4. 真正的问题：循环放在哪里、谁决定次数、状态如何稳定

把现有工作按论文时间排列，容易得到一长串名字；按架构问题排列，逻辑会清楚得多。

### 4.1 循环放在哪里？

第一种是只循环中间核心：

```text
Prelude → Recurrent Core → Recurrent Core → ... → Coda
```

[Huginn](https://arxiv.org/abs/2502.05171v2) 采用 recurrent block，让模型在 latent space 中扩展 test-time compute。论文把 proof-of-concept 扩展到 3.5B 参数、800B training tokens，并报告部分推理任务能随额外递归深度继续改善。Prelude 与 Coda 保留输入、输出角色，循环核心承担迭代计算，结构边界相对清晰。

第二种是让整个模型栈循环：

```text
F → F → F → ... → readout
```

[Ouro](https://arxiv.org/abs/2510.25741v5) 将 latent iteration、带熵正则的深度分配和大规模预训练结合起来。论文发布 1.4B 与 2.6B 模型，并报告训练数据规模达到 7.7T tokens。它证明 full-stack loop 可以扩展到更大训练规模，但这些结果来自专门的预训练流程，不等同于“把任意 pretrained Transformer 整体再跑一遍”也会有效。

第三种是按相邻层局部循环：[Loop the Loopies!](https://arxiv.org/abs/2607.16051v2) 改变了重复顺序。若普通 model-loop 是：

```text
F1 → F2 → F3 → F1 → F2 → F3
```

那么 layer-loop 更接近：

```text
F1 → F1 → F2 → F2 → F3 → F3
```

这条路线同时考虑参数复用、activation checkpointing、microbatch 和 pipeline 等系统问题。它提醒我们：循环顺序不仅改变函数表达，也改变参数局部性和并行方式。

### 4.2 谁决定循环次数？

LoopFormer 允许用户给整条 sequence 指定预算 $M$。它解决的是：同一模型如何在 1 到 $L$ 次循环之间切换。

[Mixture-of-Recursions（MoR）](https://arxiv.org/abs/2507.10524v3) 更细。它用轻量 router 给不同 token 分配递归深度，只让仍活跃的 token 继续参与后续 attention 与 MLP，并为这些 token 选择性维护 KV cache。论文在 135M 到 1.7B 范围报告了质量、模型大小和吞吐之间的 Pareto 改善。

两者不能简单地看成“粗糙版”和“高级版”：

- sequence 级预算使用规则张量，部署简单、吞吐可预测；
- token 级预算计算分配更细，但会引入 ragged batching、动态 token 集合和 cache 调度；
- 样本级 early exit 位于两者之间，适合按题目难度调整循环次数，但仍会打散 batch。

作为对照，[Mixture-of-Depths](https://arxiv.org/abs/2404.02258) 不共享层参数，而是在每一层选择 top-(k) token 进入 attention 与 MLP。它保持总预算可预测，同时动态改变计算落到哪些 token 上。它回答“每层算谁”，MoR 回答“谁继续递归”，LoopFormer 回答“整条轨迹走几步”。

### 4.3 状态为什么不会越循环越坏？

假设直接重复一个普通 residual block：

$$
h_{r+1}=h_r+G_\theta(h_r).
$$

如果 $G_\theta$ 在每轮持续产生同方向更新，状态范数可能增长；如果更新与当前任务无关，多轮以后可能漂离 pretrained representation；如果训练只监督最终一轮，中间状态也未必可读，更不能保证提前退出可用。

近期方法大致使用四类稳定器：

| 稳定器 | 代表工作 | 作用 |
| --- | --- | --- |
| 轮次或轨迹条件 | LoopFormer | 告诉共享参数“现在在哪里、这一步多大” |
| 短长路径一致性 | LoopFormer | 让短预算逼近完整预算的结果 |
| 选择性或阻尼更新 | LoopUS、Training-Free Looped Transformers | 限制每轮对已有表示的破坏 |
| loop-aware residual scaling | DeepLoop | 让残差尺度考虑同一参数被重复访问 |

[DeepLoop](https://arxiv.org/abs/2607.13491) 的出发点尤其直接：untied Transformer 的每个 residual branch 拥有独立参数，而 looped model 的同一参数会被多次访问并聚合梯度，因此只按名义深度套用普通 scaling rule 并不充分。论文在 GPT-2 small/medium 规模重新推导 scaling，并报告递归深度启用后得到改善。它提供的是稳定性证据，不足以证明同样规则已经适用于更大 LLM。

## 5. LoopFormer：把重复执行改写成一条可缩短的轨迹

LoopFormer 的核心变化可以用一行式子表示：

$$
h_i=\Phi_\theta\left(h_{i-1};t_{i-1},\Delta_i\right),
\qquad
\sum_{i=1}^{M}\Delta_i=1.
$$

$t_{i-1}$ 是当前累计的归一化时间，$\Delta_i=t_i-t_{i-1}$ 是本步跨越的区间。完整的 $L$-step trajectory 使用较细步长；较短的 $M$-step trajectory 用更少、更大的步长走完同一个 $0\rightarrow1$ 区间。

### 5.1 同一组参数为什么知道自己该做什么？

论文分别对 $t$ 与 $\Delta$ 做 embedding，再用 MLP 产生两组 RMSNorm scale 和两组 residual gate。抽象后，一个子层可以写成：

$$
x' = x + \alpha(t,\Delta)\odot
G_\theta\left(
\operatorname{RMSNorm}(x)\odot(1+\gamma(t,\Delta))
\right).
$$

这使得同一共享 block 不必在每一轮执行完全相同的更新。模型可以根据“当前位置”和“本次跨度”调整归一化后的通道尺度与残差更新强度。

重要的区分是：这里只让行为依赖轨迹条件，主 block 参数仍然共享。它不像 depth-wise LoRA 那样为每个深度保存独立的低秩参数。

### 5.2 为什么短路径也能工作？

训练时同时计算两条路径：

- 完整路径 $\mathbf{\Delta}_L$；
- 随机采样长度 $S<L$ 的 shortcut path $\mathbf{\Delta}_S$。

论文目标由完整路径 next-token loss、短路径 next-token loss和 consistency loss 组成：

$$
\mathcal L
=\mathcal L_L
+\lambda_1\mathcal L_S
+\lambda_2\mathcal L_{\mathrm{cons}}.
$$

一致性项让短路径的表示接近 stop-gradient 的完整路径目标。直观地说，短轨迹不是简单少做几步，而是在学习用“大步”近似细轨迹。

一个最小例子是：完整预算用 8 次 $1/8$ 步长，预算减半后可以用 4 次 $1/4$ 步长。两条路径最终都到达归一化时间 1；consistency training 要求后者在少一半循环时仍接近前者。

### 5.3 它解决了什么，又没有解决什么？

【来源结论】LoopFormer 论文报告模型在多个推理预算下保持可用，并随预算增加较平滑地改善。论文还观察到，同样循环次数下，不同步长 schedule 也会产生明显差异。

【证据边界】这些实验说明 trajectory conditioning 与 consistency objective 在论文设置下有效，但没有证明它们对任意已有 checkpoint 都能低成本迁移。其预算也是 sequence 级的，不会自动给困难 token 更多深度。

训练成本同样不是免费的。论文每个 batch 计算一条完整轨迹和一条随机短轨迹，因此在其设置下约为固定循环训练的 **1.5× FLOPs**；作者在 4×H100、相同 batch size、optimizer 和 data 的设置中测得约 **1.3× wall-clock slowdown**。这两个数字是论文环境的测量，不是通用部署常数。

## 6. 为什么“从 pretrained model 出发”是另一类问题

从头预训练的 recurrent model 可以让全部表示围绕循环结构共同形成。后训练改造面对的约束更苛刻：既要安装新的循环行为，又不能破坏 checkpoint 已有的语言、知识和指令能力。

### 6.1 Retrofitted Recurrence：用 curriculum 安装递归深度

[Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence](https://arxiv.org/abs/2511.07384v1) 研究如何把非递归 pretrained model 转为 depth-recurrent model。论文摘要给出的核心方法是 recurrence curriculum：训练过程中逐步增加有效深度，以较低总计算成本保持性能；其数学实验报告，在相同计算预算下，递归改造优于只继续后训练原始非递归模型。

这提供了重要的存在性证据：recurrence 不一定只能从预训练第一天开始。但它仍不意味着所有能力都能无损保留，也没有自动提供 elastic budget 或 learned exit。

### 6.2 Relaxed Recursive Transformers：共享主参数，保留深度角色

严格 weight tying 要求每个深度使用同一函数。Relaxed Recursive Transformers 引入 depth-wise LoRA：

\[
W_r=W_{\text{shared}}+B_rA_r.
\]

大矩阵 $W_{\text{shared}}$ 跨深度共享，每个递归位置只保留低秩增量。论文还用原模型权重均值初始化共享部分，并用截断 SVD 近似各深度与共享权重之间的残差。

这是一种很实用的中间态：

- rank 为 0 时，退化为严格递归共享；
- rank 增大时，模型逐渐恢复不同深度的独立角色；
- 初始化能利用原 checkpoint，而不是从随机 recurrent block 开始。

论文提出的 Continuous Depth-wise Batching 与 early exit 吞吐收益主要来自理论分析和 oracle-exit simulation；文中报告的 2–3× 不应写成已经完成端到端部署验证的速度提升。

### 6.3 LoopUS：直接把 pretrained LLM 改成 latent refinement model

[LoopUS](https://arxiv.org/abs/2605.11011v1) 将模型拆成：

```text
Encoder → Looped Reasoning Block × R → Decoder
```

其关键不是结构图，而是围绕 pretrained representation 增加了四个保护机制：

1. 根据层间 representation dynamics 选择 block decomposition；
2. 用 input-dependent selective gate，在旧状态与候选更新之间插值，缓解 hidden-state drift；
3. 用 random deep supervision 避免完整长程 BPTT 的内存开销；
4. 用 confidence head 预测何时不再需要继续 refinement。

这使 LoopUS 成为当前最贴近 Elastic-Depth Post-Training 的单篇工作之一。不过，“很适合后训练”应理解为它的任务定义与机制面向后训练，而不是已经证明它对所有模型、所有能力都无损通用。

## 7. 相关路线放到同一张表里

| 工作 | 循环位置 | 预算粒度 | 主要稳定机制 | pretrained 改造 | 关键边界 |
| --- | --- | --- | --- | --- | --- |
| Huginn | 中间 recurrent core | sequence / 固定或采样 recurrence | recurrent-depth training | 否，主要从头训练 | 超训练深度的收益不保证单调 |
| Ouro | 整个模型栈 | learned depth allocation | 专门预训练目标 | 否，主要从头训练 | 大规模结果不能直接外推到 retrofit |
| LoopFormer | 共享 block trajectory | sequence 级用户预算 | $t,\Delta t$ conditioning + consistency | 论文未验证通用 retrofit | 不是 token-adaptive |
| MoR | 共享层栈递归 | token 级 | router + selective KV cache | 通常需系统性训练 | 动态调度复杂 |
| Retrofitted Recurrence | pretrained model 中间部分 | curriculum 中的 recurrence | 逐步增加有效深度 | 是 | 未自动获得 elastic exit |
| Relaxed Recursive Transformer | 多层压成共享 block | 固定深度，可结合 exit | depth-wise LoRA + SVD init | 是 | 吞吐提升仍以模拟为主 |
| LoopUS | encoder / loop / decoder | example 级 early exit | selective gate + deep supervision | 是 | 跨模型能力保留仍需验证 |
| LOTUS | 并行 latent blocks | 固定 latent loops | gold CoT step parallel supervision | 需要特定监督 | 依赖可对齐的 reasoning steps |
| Loopie | 相邻 layer-loop | 固定 | 架构与系统协同 | 更偏专门训练 | 重点不在 adaptive depth |
| DeepLoop | 物理 block 重复 | 固定或增加 loop | loop-aware residual scaling | 未证明通用 retrofit | 验证规模较小 |
| DEQ | fixed-point operator | solver 自适应 | root finding + implicit differentiation | 工程改造困难 | solver 速度与稳定性是瓶颈 |

[LOTUS](https://arxiv.org/abs/2606.31779v2) 还提供了另一种 latent reasoning 图景：创建 $K$ 个 latent blocks，对应 gold CoT steps，并行执行 $R$ 次循环。论文报告在 3B 规模弥合 latent CoT 与 explicit CoT 的差距，并在不同形式的数学推理中将 thought-phase latency 降低 2.5–6.9×。它依赖 gold CoT-step supervision，因此与“只用终局答案安装通用循环”并不是同一任务。

[Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) 则把 weight-tied depth 推到极端：不显式指定展开层数，而是求解

$$
h^\star=F_\theta(h^\star).
$$

通过 implicit differentiation，DEQ 的 activation memory 不随有效展开深度增长。但 root solver 的收敛、速度和数值稳定性，使它更像理论参照和独立工程路线，而不是给现有 LLM 增加几次循环的直接替代。

## 8. 一个更完整的研究方案：Elastic-Depth Post-Training

将 LoopFormer 与 LoopUS 的思想组合，可以得到：

$$
\tilde h_{r+1}=F_\theta(h_r;t_r,\Delta_r),
$$

$$
h_{r+1}=(1-g_r)\odot h_r+g_r\odot\tilde h_{r+1},
$$

其中 $t_r,\Delta_r$ 控制共享 block 在不同轨迹位置的行为，$g_r$ 控制哪些通道或 token 接受本轮更新。confidence head 再估计继续循环的价值。

这套方案必须同时通过四项验收：

1. **短路径可用：** $R=1$ 不能因为训练了长路径而显著退化；
2. **长路径增益：** 从 $R=1$ 到训练最大深度，性能总体改善；
3. **外推稳定：** 超过训练深度后至少不快速崩溃；
4. **能力保留：** 改造不能只提升数学小集合，却显著破坏语言建模和通用任务。

### 8.1 路线 A：Nanbeige4.2 + trajectory consistency

[Nanbeige4.2-3B](https://arxiv.org/abs/2607.22083v2) 从头使用 Looped Transformer 预训练；其公开描述确认 layer stack 被复用以增加有效深度。[一项独立部署研究](https://arxiv.org/abs/2608.13987v1)进一步将该结构描述为对 layer stack 的第二次 forward pass。后一个来源主要研究 Apple Silicon 部署问题，因此这里只用它确认执行结构，不借用其任务效果结论。

因此它适合回答：

> 固定两遍预训练形成的 Looped LLM，能否仅通过后训练获得 elastic depth？

最小改造包括：

- 给循环加入 $t,\Delta t$ conditioning；
- residual gate 使用近似 identity-preserving 的初始化；
- 同时训练完整两遍和随机短轨迹；
- 测试 $R=1,2,3,4$，但把 $R>2$ 明确视为外推区间。

这条路线复用现成 recurrent checkpoint，但需要自建 trajectory-conditioned block、双轨迹训练流程、跨深度评测和能力保持验证。现有 checkpoint 不是完整实验合同。

### 8.2 路线 B：Qwen + Relaxed Recursive LoRA

从普通 Qwen checkpoint 出发，把连续若干中间层压成共享 recurrent block，并为每个递归深度保留独立 LoRA：

$$
F_{\theta+\Delta\theta_r}.
$$

建议至少做四组对照：

- 原始 Qwen continual post-training；
- rigid weight tying；
- depth-wise LoRA；
- depth-wise LoRA + early exit。

这比只比较 Qwen 与 Nanbeige 更干净，因为它在同一初始 checkpoint 上隔离“严格共享”和“按深度保留小参数”带来的影响。

### 8.3 路线 C：LoopFormer + LoopUS

这是创新空间最大、组件也最多的一条路线：

- LoopFormer 提供全局预算与 trajectory consistency；
- LoopUS 提供 selective refinement、deep supervision 与 confidence exit；
- depth-wise LoRA 可作为严格共享失败时的松弛项；
- DeepLoop-style scaling 可作为深循环不稳定时的对照，而不是默认全部叠加。

核心研究问题不是“组合后分数是否更高”，而是三个组件分别解决了什么：

- consistency 是否主要改善短路径？
- gate 是否主要改善长路径稳定性？
- confidence exit 是否真的能按难度节省平均计算，而不是把固定阈值包装成自适应？

## 9. 最小可证伪实验

如果资源有限，我会先做一个能在两周内推翻错误方向的实验，而不是一开始训练完整系统。

### 阶段一：确认循环改造没有立即破坏模型

固定一个小规模 pretrained backbone，只改中间 block。比较：

| 组别 | 共享方式 | 条件输入 | consistency | gate |
| --- | --- | --- | --- | --- |
| A | 不共享 | 无 | 无 | 无 |
| B | rigid tie | 无 | 无 | 无 |
| C | rigid tie | $t,\Delta t$ | 有 | 无 |
| D | rigid tie | $t,\Delta t$ | 有 | 有 |
| E | depth-wise LoRA | $t,\Delta t$ | 有 | 有 |

先看 validation perplexity、通用任务保持率和 $R=1,2,3,4$ 的完整曲线。只汇报最佳深度没有意义，因为 elastic model 的研究对象正是整条 compute–quality curve。

### 阶段二：检查“更多计算”是否真的更好

对每个样本记录：

$$
\Delta \ell_r=\ell(h_{r+1})-\ell(h_r),
$$

以及 hidden-state norm、相邻轮 cosine distance、logit entropy 与输出是否翻转。理想状态不是每个样本都严格单调，而是困难样本能从更多循环中获得更稳定的期望增益。

若以下任一情况出现，就应该判定当前方案失败：

- $R=1$ 相对 base checkpoint 大幅退化；
- 训练深度内性能没有随预算形成可辨认趋势；
- gate 几乎恒为 0 或 1；
- exit score 与“继续一轮后的真实收益”不相关；
- 只在训练格式内改善，换任务后循环立刻失效。

### 阶段三：最后才测系统收益

只有模型层结果成立后，才值得实现动态 batch、KV cache 和 early-exit scheduler。系统评测至少同时报告：

- 平均 executed FLOPs；
- tokens/s 与端到端 latency；
- 峰值显存；
- 不同输入长度和 batch size；
- quality-matched 与 compute-matched 两套比较。

否则“参数量下降”“平均循环次数下降”都不能自动证明部署更快。

## 10. Claim–evidence map

| Claim | 直接证据 | 强度 | 不能外推成什么 |
| --- | --- | --- | --- |
| Looped model 可以用递归深度扩展 latent compute | Huginn 在 3.5B、800B tokens 设置中的实验 | 中到强 | 任意增加循环都会持续变好 |
| 同一模型可以学习多个全局计算预算 | LoopFormer 的 shortcut conditioning 与多预算实验 | 强，限论文设置 | 已实现 token-adaptive compute |
| token 可学习不同递归深度 | MoR 的 router、selective attention/KV cache 实验 | 强，限 135M–1.7B | 大模型部署调度已经无成本解决 |
| pretrained model 可以被改造成 recurrent model | Retrofitted Recurrence、RRT、LoopUS | 中到强 | 所有原能力均能无损保留 |
| depth-wise LoRA 能缓解严格共享限制 | RRT 的初始化、uptraining 与消融 | 强，限其模型设置 | 理论吞吐提升已完成真实系统验证 |
| loop-aware residual scaling 有必要 | DeepLoop 的推导与 GPT-2 规模实验 | 中 | 已验证大规模 LLM 的统一 scaling law |
| LoopFormer + LoopUS 可能形成完整 elastic-depth retrofit | 跨论文机制综合 | 假设 | 已被任何论文直接验证 |

## 11. 局限与仍未回答的问题

第一，**latent compute 不透明**。显式 CoT 可以检查中间步骤，普通 looped hidden state 很难判断某一轮到底完成了什么操作。LOTUS 借助 gold CoT-step supervision 提高对齐性，但也改变了数据要求。

第二，**深度外推不是默认能力**。训练看过 2、4、8 次循环，不代表 16 次仍然有效。共享参数只保证函数可以继续调用，不保证状态仍在训练分布内。

第三，**自适应停止缺少天然标签**。confidence head 可以预测当前答案置信度，但“当前有信心”不等于“再算一轮没有价值”。更合适的监督信号可能是 marginal improvement，而不仅是当前正确率。

第四，**质量曲线与系统曲线可能冲突**。token-level routing 节省理论计算，却可能因动态索引、低硬件利用率和 cache 搬运损失真实吞吐。早退也可能使 batch 内同步变差。

第五，**后训练的数据会决定循环学成什么**。如果只有终局答案，模型可能学到重复确认；如果有过程监督，模型可能把每一轮对齐到步骤，但也可能过拟合某种 reasoning format。数据合同与架构同样需要单独设计。

## 12. 一句话带走

> Looped Transformer 的真正下一步，不是证明同一组参数还能再执行一次，而是让每一次额外执行都有可预测的边际价值，并让模型知道何时值得继续、何时应该停止。

## 参考资料

- [LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation](https://arxiv.org/abs/2602.11451v1)，v1，2026-02-11。
- [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171v2)，v2。
- [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741v5)，v5。
- [Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation](https://arxiv.org/abs/2507.10524v3)，v3。
- [LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models](https://arxiv.org/abs/2605.11011v1)，v1。
- [Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence](https://arxiv.org/abs/2511.07384v1)，v1。
- [Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA](https://arxiv.org/abs/2410.20672v3)，v3。
- [Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers](https://arxiv.org/abs/2606.31779v2)，v2。
- [Loop the Loopies!](https://arxiv.org/abs/2607.16051v2)，v2。
- [DeepLoop: Depth Scaling for Looped Transformers](https://arxiv.org/abs/2607.13491)，访问于 2026-08-18。
- [Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models](https://arxiv.org/abs/2404.02258)。
- [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710)。
- [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377)。
- [Nanbeige4.2-3B: Unlocking Agentic Capabilities in a Compact Model](https://arxiv.org/abs/2607.22083v2)，v2。
- [Nanbeige4.2-3B on Apple Silicon: Fixing Deployment Bugs and Decreasing Looped Transformer Memory Overhead](https://arxiv.org/abs/2608.13987v1)，v1；仅用于补充执行结构与部署边界。

