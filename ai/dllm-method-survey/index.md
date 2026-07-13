# DLLM 方法综述：从掩码扩散到大语言扩散模型


如果把 `DLLM` 理解成 `Diffusion Language Model` 或 `Diffusion Large Language Model`，那它现在最值得讨论的地方不是“它能不能一次生成多个 token”这么简单。

更准确地说，DLLM 在重新打开一个被自回归 LLM 几乎盖住的问题：

**语言模型一定要从左到右写下去吗？**

自回归模型的答案是：是的，把序列概率拆成一串 next-token 条件概率，训练、推理、缓存、服务系统全都围绕这个假设展开。扩散语言模型的答案是：未必。我们也可以先拿到一个被破坏的句子，再让模型用双向上下文一步步修复它；也可以先生成骨架，再填细节；也可以在多个位置同时做决策，只把足够确定的位置提交。

所以 DLLM 真正有意思的地方，是它把语言生成从“写作”改成了“编辑”。

<!--more-->

## 如果只看三句话

- DLLM 的核心不是简单替代 AR，而是把生成过程改成“加噪-去噪-迭代提交”：训练时学会从不同噪声强度恢复原文，推理时从全 mask 或部分噪声状态逐步还原文本。
- 这条线经历了三次方法迁移：先在连续 embedding 空间做扩散，再在离散 token 空间做 D3PM/SEDD，最后收敛到更容易放大的 masked diffusion / block diffusion。
- 2025 之后的关键问题已经从“能不能训出语言模型”变成“能不能像 AR LLM 一样规模化、后训练、缓存、服务和评测”。

## 1. 从 AR 到 DLLM：建模假设变了什么

自回归语言模型把一个长度为 $L$ 的序列 $x=(x_1,\dots,x_L)$ 分解成：

$$p(x)=\prod_{i=1}^{L}p(x_i\mid x_{<i})$$

这条公式非常强，因为它带来三个工程红利。

- 训练目标简单，就是 next-token cross entropy。
- 推理过程自然，就是一个 token 一个 token 往后采样。
- KV cache 极其好用，因为新 token 只会追加到右边，旧 token 的 hidden states 不需要重算。

DLLM 选择的路不同。它不直接建模从左到右的条件概率，而是定义一个前向破坏过程 $q(x_t\mid x_0)$：从干净文本 $x_0$ 出发，逐渐把它变成更噪的 $x_t$。然后训练一个反向模型 $p_\theta(x_{t-\Delta t}\mid x_t)$ 或 $p_\theta(x_0\mid x_t)$，让它学会从噪声状态恢复文本。

也就是说，AR 的基本动作是：

```text
prefix -> next token
```

DLLM 的基本动作是：

```text
corrupted sequence -> less corrupted sequence
```

这带来的最大差异是注意力方向。AR 通常只能看左侧 prefix；DLLM 可以看见左右两边的上下文，因为目标不是预测“下一个位置”，而是修复“当前被破坏的位置”。这也是 DLLM 在 infilling、任意顺序生成、反转关系、局部编辑和多 token 并行提交上天然更顺手的原因。

但代价也很直接：AR 每一步只跑一个增量 token，DLLM 每一步通常要处理一整段序列。理论并行性不等于真实服务吞吐，后面所有加速工作几乎都在补这个账。

## 2. 最早的一条路：连续 embedding diffusion

把扩散模型搬到文本上的第一个自然想法是：既然 DDPM 在连续图像空间很好用，那就把 token 映射成连续 embedding，在 embedding 上加高斯噪声。

连续扩散的标准前向过程可以写成：

$$z_t=\sqrt{\alpha_t}z_0+\sqrt{1-\alpha_t}\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)$$

训练时，模型从 $z_t$ 预测噪声、干净 embedding 或 score；推理时，从高斯噪声逐步去噪，最后再把连续向量映射回离散 token。

`Diffusion-LM` 和 `DiffuSeq` 大体属于这条路线。它们的价值不在于今天看起来有多像大模型，而在于证明了一个关键点：扩散过程的中间状态是连续、可微、可被外部目标引导的。比如控制情感、句法结构、关键词、源句条件时，可以在中间 latent 上加梯度，让生成往目标方向走。

这条路线的优点是：

- 可以直接复用连续扩散模型的数学工具；
- 中间变量可微，适合做复杂可控生成；
- 对 seq2seq、改写、摘要这类条件生成任务很自然。

它的硬伤也很明显：

- token 到 embedding 再回 token 有离散化 gap；
- 语言模型最终要输出词表分布，连续空间的好轨迹未必对应自然文本；
- 很难像 AR LLM 那样直接吃超大语料、超大词表和成熟训练栈。

所以连续 embedding diffusion 更像是 DLLM 的第一阶段：它证明“扩散可以做文本”，但还没有回答“扩散怎么成为大语言模型”。

## 3. 第二条路：离散扩散，把噪声放回 token 空间

更彻底的做法是：不要绕到 embedding 空间，直接在 token 状态上定义扩散。

设词表为 $\mathcal{V}$，某个位置的 token 是 $x_t\in\mathcal{V}$。离散扩散用一个转移矩阵 $Q_t$ 描述 token 如何被破坏：

$$q(x_t\mid x_{t-1})=\mathrm{Cat}(x_t; Q_t x_{t-1})$$

D3PM 的重要性就在这里。它说明离散扩散不必只是“随机换成任意 token”，转移矩阵可以有结构：可以是 uniform corruption，可以根据 embedding 邻近关系替换，也可以引入吸收态 `[MASK]`。其中 `[MASK]` 这条分支后来变得极其重要，因为它把扩散语言模型和 BERT 式 masked language model 接了起来。

如果前向过程是吸收态 masking，那么一个 token 一旦被 mask，就保持 mask。用一个噪声强度 $t$ 表示 mask 概率，可以近似写成：

$$q(x_t^i=M\mid x_0^i)=t,\quad q(x_t^i=x_0^i\mid x_0^i)=1-t$$

这时反向过程就变成：

```text
看到一个部分 mask 的句子
预测每个 mask 位置原来是什么 token
选择若干位置提交
继续迭代
```

这一步非常关键。因为从这里开始，DLLM 不再是“把图像扩散公式硬搬到文本”，而是在离散语言结构里找到自己的自然形式。

## 4. SEDD：不预测 token，而是预测概率比

离散扩散还有一条更偏理论的路线：能不能把连续 score matching 的思想迁移到离散空间？

连续扩散里的 score 大致是 $\nabla_x\log p_t(x)$，它告诉模型“往哪个方向移动，数据概率会上升”。但离散 token 没有连续梯度，所以 SEDD 换了一个对象：不预测梯度，而预测把当前 token 替换成另一个 token 后的概率比。

可以把离散 score 理解成：

$$s_\theta(x,t,y)\approx \frac{p_t(x^{i\rightarrow y})}{p_t(x)}$$

其中 $x^{i\rightarrow y}$ 表示把序列 $x$ 的第 $i$ 个位置替换成 token $y$。如果这个比值大，说明替换后更像数据分布；如果小，说明替换方向不靠谱。

这比“直接预测原 token”更像扩散模型的本体：模型学习的是数据分布在离散邻域里的相对形状。SEDD 用 score entropy loss 训练这个 ratio estimator，在 GPT-2 量级的语言建模上把离散扩散的困惑度和生成质量往前推了一大步。

但它也暴露出一个现实问题：方法再漂亮，最后还要进入 LLM 工程系统。词表很大、序列很长、采样步数很多时，ratio-based 建模的复杂性会变成工程负担。于是更简单、更像 MLM 的 masked diffusion 变成了后续大规模 DLLM 的主线。

## 5. MDLM：为什么 masked diffusion 成了主干

`MDLM` 的核心判断很朴素：masked diffusion 本身已经足够强，关键是把目标、参数化和采样做干净。

训练时，从真实文本 $x_0$ 采样一个噪声强度 $t$，按比例 mask 掉一部分 token，得到 $x_t$。模型看到 $x_t$，只在 mask 位置预测原 token：

$$\mathcal{L}(\theta)=\mathbb{E}_{t,x_0,x_t}\left[w(t)\sum_{i:x_t^i=M}-\log p_\theta(x_0^i\mid x_t)\right]$$

这看起来很像 BERT 的 MLM，但差别在两个地方。

第一，mask 比例不是固定的 15%，而是覆盖从“几乎没坏”到“几乎全坏”的连续噪声强度。模型因此学到不同修复难度下的 denoising 行为。

第二，它不是只为了学表示，而是有明确的生成过程。推理时可以从全 mask 开始，每一步预测所有 mask 位置，然后按置信度或 schedule 解开一部分位置：

```text
[M] [M] [M] [M] [M]
-> 今天 [M] [M] [M] [M]
-> 今天 天气 [M] [M] [M]
-> 今天 天气 很 [M] [M]
-> 今天 天气 很 好 。
```

MDLM 的意义在于，它把 masked LM、离散扩散和可采样生成模型放进了同一个框架里。BERT 以前也会预测 mask，但它通常不是一个完整的生成模型；MDLM 则把“从全噪声到文本”的概率过程补齐了。

这也是为什么后来的 LLaDA、Dream、iLLaDA 等大模型都更接近 masked diffusion：它简单、稳定、可扩展，也能直接复用 Transformer。

## 6. 大模型化：LLaDA、Dream 和 iLLaDA 做对了什么

2025 年之后，DLLM 的问题从“能不能做语言建模”变成了“能不能做大语言模型”。

`LLaDA` 是一个标志性节点。它用一个 vanilla Transformer 做 mask predictor，不使用 causal mask，而是用全双向注意力从 masked sequence 里恢复 token。它的预训练目标可以写成：

$$\mathcal{L}(\theta)=-\mathbb{E}_{t,x_0,x_t}\left[\frac{1}{t}\sum_{i=1}^{L}\mathbf{1}[x_t^i=M]\log p_\theta(x_0^i\mid x_t)\right]$$

这里的 $\frac{1}{t}$ 可以理解成对不同 mask 比例的归一化，避免高 mask 比例样本因为 mask token 更多而主导 loss。LLaDA 的核心贡献不是发明一个复杂结构，而是证明 masked diffusion 可以按 LLM 的 pretraining + SFT 范式放大到 8B 级别，并在 in-context learning、instruction following、反转关系任务上展现出和 AR 模型不一样的能力形态。

`Dream 7B` 往前推了一步。它仍然是离散扩散/迭代去噪，但强调用 AR checkpoint 初始化、上下文自适应的 token-level noise rescheduling，以及更灵活的质量-速度 trade-off。这里有一个很重要的趋势：DLLM 不一定每次都要从零训练。AR LLM 已经积累了大量数据、tokenizer、架构和权重资产，把这些资产“diffusify” 可能比纯从头训练更经济。

到 2026 年的 `iLLaDA`，重点又变成了 scale recipe：更大规模预训练、更长 instruction tuning、更合理的变长生成和评测打分。它说明 DLLM 的上限至少还没有被简单方法摸到。也就是说，今天看 DLLM，不应该只问“它现在有没有超过最强 AR 模型”，而应该问：

**当 AR 的训练工程红利被逐步迁移到 diffusion 之后，哪些能力会因为双向生成和迭代修复而变得不同？**

## 7. Block Diffusion：向 AR 借系统能力

全序列 masked diffusion 有一个麻烦：它擅长任意位置修复，但不天然支持任意长度生成和 AR 式 KV cache。

Block Diffusion 的思路很工程，也很聪明：序列之间仍然按 block 自回归推进，但每个 block 内部用 diffusion 并行生成。

可以把序列切成 $B$ 个 block：

$$p(x)=\prod_{b=1}^{B}p_\theta(x^{(b)}\mid x^{(<b)})$$

其中每个 $p_\theta(x^{(b)}\mid x^{(<b)})$ 不是从左到右生成 block 内 token，而是对当前 block 做 masked diffusion。

这等于在两个极端之间插了一个旋钮：

- block size = 1 时，接近 AR；
- block size = 全序列时，接近 full diffusion；
- 中间 block size 则在 KV cache、并行 token 提交、变长生成之间折中。

这条线非常重要，因为它承认了一个现实：AR 的系统栈太成熟了，DLLM 如果完全不兼容 KV cache 和流式服务，就很难在真实生产里拿到理论并行性的好处。Block Diffusion、Multi-Block Diffusion、Set Diffusion 这些后续工作，本质上都在调这个旋钮：一次生成多少位置、这些位置是否连续、能否复用 prefix cache、能否支持更灵活的长度和顺序。

## 8. 推理：DLLM 的瓶颈不在“并行”，而在“提交策略”

很多人第一次听 DLLM，会立刻想到“并行生成，所以更快”。这句话只说对了一半。

DLLM 的一次 forward 确实可以预测很多位置，但问题是：预测了不等于能提交。过早提交低置信 token，会让后续上下文变坏；太保守，又会浪费 forward 次数。

所以 DLLM 推理的核心变量不是“每步预测多少 token”，而是：

```text
哪些 token 已经稳定到可以提交？
哪些 token 需要继续被 mask，等更多上下文形成后再决定？
```

最基本的采样过程大概是：

```text
1. 给定 prompt，把 response 区域初始化为 [MASK]
2. 第 k 步，模型输出所有 masked 位置的分布
3. 根据置信度、位置 schedule 或稳定性选择一批 token
4. 提交这些 token，其他位置保持 mask
5. 重复直到没有 mask，或达到最大步数
```

`Fast-dLLM` 的 confidence-aware parallel decoding，就是只提交超过置信度阈值的位置，避免“为了并行而并行”。`LESS` 进一步把提交看成在线停止问题：不只看 top-1 概率高不高，还看 top-1 是否跨多步稳定、分布是否稳定。`SAID` 则把 token 分成 scaffold 和 detail，先多花计算确定语义骨架，再用更少步数补细节。

这些方法共同说明一件事：DLLM 的推理不是简单的采样问题，而是一个 **compute allocation** 问题。每个 forward 都是一笔预算，应该花在最不确定、最影响全局结构的位置上。

## 9. 缓存与服务：为什么 DLLM 不能直接套 AR KV cache

AR 模型的 KV cache 好用，是因为生成第 $i$ 个 token 时，前面 token 的表示不会因为未来 token 变化而改变。DLLM 不一样。它有双向注意力，一个位置从 mask 变成具体 token，会影响其他所有位置的表示。

这意味着：精确 KV cache 在 full diffusion 里并不成立。

但这并不代表不能缓存。`dKV-Cache`、`Fast-dLLM`、`ES-dLLM`、`Dynamic-dLLM`、`Sangam` 等工作抓住了一个经验事实：相邻 denoising step 之间，很多 token 的 hidden states / KV states 变化并不剧烈。于是可以做近似缓存、延迟刷新、重要性跳过、动态 cache budget，或者把 dLLM 服务拆成类似 AR 系统里的 recurring prefill/decode 结构。

这里的关键 trade-off 是：

```text
缓存越激进，速度越快，但表示越 stale；
刷新越频繁，质量越稳，但越接近原始慢推理。
```

所以 DLLM 真正落地时，方法论文里的“采样步数”还不够描述成本。更完整的成本模型至少要看：

- 每步是否全序列 forward；
- 每次提交多少 token；
- cache 多久刷新一次；
- block size 多大；
- prefill 和 decode worker 怎么调度；
- 长序列下 attention 复杂度如何增长。

这也是为什么 2026 年开始出现很多服务系统工作。DLLM 要和 AR LLM 竞争，不能只在 benchmark 上说“同等步数更好”，还要在吞吐、延迟、显存、batching、流式输出上进入同一个竞技场。

## 10. 后训练：SFT 和 RL 也要重新写一遍

DLLM 的 post-training 不能简单照搬 AR。

AR SFT 的训练目标是：

$$-\sum_i\log p_\theta(y_i\mid x,y_{<i})$$

但 masked diffusion SFT 通常是：prompt 保持可见，response 随机 mask，模型预测被 mask 的 response token。于是训练时模型会看到右侧上下文，推理时却可能按 block 或置信度逐步生成。如果训练的 mask 分布和推理的生成状态不一致，就会出现 train-inference mismatch。

`Blockwise SFT` 这类方法就在修这个问题：把 response 分块，只对当前 active block 加 mask，前面的 block 固定可见，后面的 block 完全隐藏，让训练状态更像 block-wise inference。

RL 更麻烦。AR 里我们可以把一条 response 的 logprob 写成 token logprob 之和，PPO/GRPO/DPO 都围绕这个分解做。DLLM 的生成是一条多步去噪轨迹，最终文本的概率往往要通过 ELBO、pseudo-likelihood 或均值场近似估计，importance ratio 会更 noisy。

`d1` 提出的 `diffu-GRPO` 是这条线的早期代表：先做 masked SFT，再把 GRPO 风格的 critic-free policy gradient 适配到 dLLM。后续 StableDRL、AGRPO、GDPO、DARE 等工作继续处理 ratio 估计、梯度方差、reward collapse 和统一执行栈。

这里有个很值得记住的判断：

**DLLM 的 RL 难点不只是奖励函数，而是“动作”到底是什么。**

在 AR 里，一个动作通常是下一个 token；在 DLLM 里，一个动作可以是某一步对若干 mask 位置的预测、一次提交策略、甚至一整条去噪轨迹。动作定义不同，logprob、KL、ratio、credit assignment 全都会变。

## 11. 方法对比：几条主线怎么取舍

| 方法线 | 代表工作 | 核心建模对象 | 优点 | 主要问题 |
| :--- | :--- | :--- | :--- | :--- |
| 连续 embedding diffusion | Diffusion-LM, DiffuSeq | 连续 token embedding | 可微、可控、适合条件生成 | 离散化 gap，难规模化到 LLM |
| D3PM 离散扩散 | D3PM | token 转移矩阵 | 理论清楚，可设计 corruption 结构 | 大词表和采样复杂 |
| Score/ratio 离散扩散 | SEDD | 离散邻域概率比 | 更接近 score matching，语言建模质量强 | 工程实现和大规模化更复杂 |
| Masked diffusion | MDLM, LLaDA, Dream, iLLaDA | mask token 的恢复分布 | 简单、稳定、可复用 Transformer，适合放大 | 固定长度、采样步数、缓存困难 |
| Block / Set diffusion | BD3LM, MultiBD, Set Diffusion | block 或 token set 的扩散条件分布 | 支持变长、KV cache、半自回归服务 | 需要处理 block 内外训练-推理对齐 |
| 推理加速 | Fast-dLLM, dKV-Cache, LESS, SAID, ES-dLLM, Sangam | 提交策略、缓存、调度 | 把理论并行转成真实吞吐 | 速度-质量 trade-off 复杂 |
| 后训练 | d1, diffu-GRPO, StableDRL, DARE | denoising trajectory 或近似 likelihood | 推动 reasoning/alignment | ratio 估计和 credit assignment 尚未稳定 |

如果只给一个实践取舍，我会这样看：

- 想研究可控生成和编辑，连续 diffusion 和 guided discrete diffusion 仍然很有价值。
- 想做通用 LLM，当前主线是 masked diffusion / block diffusion。
- 想做部署，重点不是再发明一个 denoising loss，而是缓存、提交策略和 serving scheduler。
- 想做 reasoning，必须认真处理 DLLM 自己的动作空间和概率估计，直接套 AR-RL 很容易不稳。

## 12. DLLM 真正可能赢在哪里

DLLM 的优势不应该被压缩成“更快”。更快当然重要，但现在看，它真正有差异的地方至少有四个。

第一是双向条件。模型生成某个位置时可以同时看左边和右边，这对 infilling、结构化补全、代码编辑、表格/JSON 修复、长文本局部改写很自然。

第二是任意顺序生成。AR 必须沿着时间轴走，DLLM 可以先定关键位置，再补细节。这对 planning 任务很有想象力：先生成结论骨架、变量、函数签名、段落结构，再逐步细化。

第三是质量-速度可调。同一个模型可以用少步数快速出草稿，也可以用更多 denoising step 提高质量。这和 AR 里“多采几条再 rerank”不完全一样，因为 DLLM 的 compute 是花在同一条输出的 refinement 上。

第四是编辑式交互。用户如果给出部分文本、部分约束、部分空洞，DLLM 不需要把它们硬转成 prefix，而是可以直接把已知位置固定住，生成未知位置。这更接近很多真实写作和编程工作流。

## 13. 但 DLLM 也有几个还没解决的硬问题

第一，开放式长文本生成还没有完全证明自己。AR 的 left-to-right inductive bias 虽然看起来笨，但非常适合叙事推进和因果链展开。DLLM 的任意顺序如果没有好 schedule，可能会过早锁定错误骨架，后面只能局部修补。

第二，评测仍然不够公平。DLLM 的表现对 denoising steps、block size、mask schedule、confidence threshold、是否允许变长、是否启用 cache 非常敏感。如果只报一个 accuracy，不说明推理预算，信息是不完整的。

第三，系统复杂度还在快速变化。AR LLM 的 serving stack 已经被工业界反复打磨，而 DLLM 的缓存和调度仍然是研究热点。今天的“快”可能依赖某个特定 batch、长度、硬件和近似策略。

第四，后训练理论还不稳。尤其是 RL：最终文本奖励怎么分配到多步去噪动作？KL 约束到底约束哪一个分布？sequence probability 用什么近似？这些问题不解决，DLLM reasoning 的提升就容易变成 recipe search。

第五，生态还年轻。tokenizer、chat template、streaming、tool calling、function calling、structured output、safety filter、评测 harness，AR 世界里的每个默认组件，在 DLLM 里都可能要重新适配。

## 14. 我对这条线的判断

我不觉得 DLLM 会在短期内把 AR LLM 整体替换掉。AR 不是偶然胜出的，它拥有极其强的概率分解、训练简洁性和系统惯性。

但我也不觉得 DLLM 只是一个“看起来像扩散的 BERT”。更准确的判断是：

**DLLM 是一种把语言生成从 next-token prediction 推向 iterative refinement 的范式。**

这件事一旦成立，它就不一定要在所有任务上正面打败 AR 才有价值。它可以先在这些场景里赢：

- infilling 和局部编辑；
- 代码补全与代码修复；
- 结构化输出修复；
- 多候选并行 refinement；
- 需要先规划后填充的 reasoning；
- 高吞吐、短回答、可批量提交 token 的服务场景；
- 多模态理解中需要双向对齐和局部重建的任务。

长远看，最可能留下来的也许不是纯 AR 或纯 diffusion，而是中间形态：block-wise、set-wise、semi-autoregressive、cache-aware 的混合模型。它们会保留 AR 的系统优势，又吸收 diffusion 的并行修复和双向条件。

如果用一句话总结 DLLM 的方法史，我会这样写：

**早期 DLLM 在问“扩散能不能生成文本”；现在 DLLM 在问“语言模型能不能从写作机器变成编辑机器”。**

这个问题值得继续追。

## 参考链接

- D3PM: [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006)
- Diffusion-LM: [Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217)
- DiffuSeq: [DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models](https://arxiv.org/abs/2210.08933)
- SEDD: [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834)
- MDLM: [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)
- SMDM: [Scaling up Masked Diffusion Models on Text](https://arxiv.org/abs/2410.18514)
- LLaDA: [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)
- Block Diffusion: [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](https://arxiv.org/abs/2503.09573)
- d1 / diffu-GRPO: [Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning](https://arxiv.org/abs/2504.12216)
- dKV-Cache: [dKV-Cache: The Cache for Diffusion Language Models](https://arxiv.org/abs/2505.15781)
- Fast-dLLM: [Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](https://arxiv.org/abs/2505.22618)
- Dream 7B: [Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487)
- dLLM framework: [dLLM: Simple Diffusion Language Modeling](https://arxiv.org/abs/2602.22661)
- DARE: [Diffusion Large Language Models Alignment and Reinforcement Executor](https://arxiv.org/abs/2604.04215)
- iLLaDA: [Improved Large Language Diffusion Models](https://arxiv.org/abs/2606.25331)
- Sangam: [Efficiently Serving Diffusion LLMs with the AR Stack](https://arxiv.org/abs/2607.04206)
- 综述: [A Survey on Diffusion Language Models](https://arxiv.org/abs/2508.10875)
- 综述: [Discrete Diffusion in Large Language and Multimodal Models: A Survey](https://arxiv.org/abs/2506.13759)

