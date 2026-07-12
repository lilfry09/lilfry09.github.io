# 读 Qwen Bebop 与 DeepSeek DSpark：Speculative Decoding 正在从技巧变成系统工程


最近两篇关于 `speculative decoding` 的论文很值得放在一起看。

一篇是 Qwen Team 的 **Bebop**，全名是 *Breaking Entropy Bounds: Accelerating RL Training via MTP with Rejection Sampling*。它关心的是：在大规模 RL 后训练里，rollout 太慢，能不能用 `MTP` 把采样阶段加速起来？

另一篇是 DeepSeek 的 **DSpark**，全名是 *Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation*。它关心的是：在线上高并发 serving 里，speculative decoding 为什么经常一上负载就不稳，怎样让它真正移动吞吐-延迟前沿？

这两篇看起来一个讲训练，一个讲推理；一个讲 Qwen 的 RL pipeline，一个讲 DeepSeek-V4 的线上服务。但我读完之后觉得，它们其实在回答同一个更大的问题：

**speculative decoding 的核心已经不只是“多猜几个 token”，而是怎样让猜测、验证、分布匹配和系统负载一起闭环。**

<!--more-->

## 如果只看三句话

- Qwen Bebop 的重点是：RL 训练时策略熵会上下波动，传统 MTP 接受率会被熵压住；要用 `rejection sampling + e2e TV loss` 直接优化接受率。
- DeepSeek DSpark 的重点是：线上服务里长 draft block 不一定更快，因为低置信 suffix 会浪费 target model 的 batch capacity；要用置信度调度决定每个请求验证多长。
- 两篇论文共同说明：speculative decoding 正在从模型侧的小技巧，变成一套横跨训练目标、draft 架构、验证算法和 serving scheduler 的系统工程。

## 先把 speculative decoding 的骨架说清楚

自回归模型慢，是因为每生成一个 token 都要跑一次大模型 forward。`speculative decoding` 的基本想法是：

```text
小 drafter 先猜一串 token
大 target model 一次 forward 并行验证这一串
能接受多少就吃多少
第一个拒绝的位置由 target model 修正
```

如果 draft 足够便宜、接受 token 足够多，平均每个 token 摊到的大模型 forward 次数就下降了。

所以所有 speculative decoding 方法都绕不开三个变量：

- draft 要足够快；
- draft 和 target 的分布要足够接近；
- 验证的 token 数不能浪费太多系统容量。

早期我们很容易把重点放在第一个变量上：让 drafter 小一点、快一点。但 Bebop 和 DSpark 都在强调后两个变量。真正难的不是“猜”，而是：

```text
猜出来的 token 在当前分布下会不会被接受；
这些 token 在当前系统负载下值不值得验证。
```

这就是两篇论文的共同底色。

## 用公式看：到底什么叫“猜得值”

先设定两个分布。

在当前位置，target model 的下一个 token 分布是：

$$
p(y) = P_{\text{target}}(y \mid x, y_{<t})
$$

drafter 给出的分布是：

$$
q(y) = P_{\text{draft}}(y \mid x, y_{<t})
$$

如果 drafter 采样出一个候选 token $\hat y \sim q$，标准 rejection sampling 的接受概率是：

$$
\alpha(\hat y) = \min \left(1, \frac{p(\hat y)}{q(\hat y)}\right)
$$

把 $\hat y$ 对 $q$ 取期望，就得到单步平均接受率：

$$\alpha_{\text{RS}} = \mathbb{E}_{\hat y \sim q}\left[\min\left(1, \frac{p(\hat y)}{q(\hat y)}\right)\right] = \sum_y \min(p(y), q(y))$$

而 total variation distance 定义为：

$$d_{\text{TV}}(p, q) = \frac{1}{2}\sum_y |p(y)-q(y)|$$

所以有一个非常关键的等式：

$$
\alpha_{\text{RS}} = 1 - d_{\text{TV}}(p, q)
$$

这就是为什么 Bebop 和 DSpark 都会反复围绕 `TV`、`accept length`、`confidence` 做文章。因为 speculative decoding 的收益，不是由 KL、CE 或困惑度直接决定，而是由 draft 分布和 target 分布的**重叠面积**决定。

如果一次 draft $\gamma$ 个 token，记每一步条件接受概率为 $\alpha_k$，那么至少接受前 $j$ 个 token 的概率近似是：

$$
P(\tau \ge j) = \prod_{k=1}^{j} \alpha_k
$$

期望接受长度可以写成：

$$\mathbb{E}[\tau] = \sum_{j=1}^{\gamma} P(\tau \ge j) = \sum_{j=1}^{\gamma}\prod_{k=1}^{j}\alpha_k$$

这条式子也解释了一个常见现象：后面 token 的质量再高，如果前面第一个低置信 token 被拒了，后面的 token 全部作废。所以 speculative decoding 的优化重点，天然会偏向 prefix，而不是平均对待所有 draft position。

## Qwen Bebop：RL 里的瓶颈不是 MTP 不会猜，而是策略熵在变

Bebop 的场景非常具体：大模型 RL 后训练。

现代 reasoning / coding / agentic RL 里，rollout 往往是最贵的阶段。模型要在当前 policy 下生成很长的轨迹，有时还要多轮工具调用、沙箱执行、代码测试。哪怕 update 本身已经异步化，rollout 仍然会拖住整条训练流水线。

所以一个自然想法是：既然推理时可以用 MTP 加速，那 RL rollout 也能不能用 MTP 加速？

论文的回答是：能，但直接套会坏。

传统解释通常会说，RL 过程中 policy 权重一直在变，frozen MTP head 跟不上，所以 draft-target 分布 mismatch 变大，接受率下降。Bebop 的关键发现更细一点：**真正主导接受率波动的，不是权重更新带来的 mismatch，而是 target policy 的熵。**

RL 为了探索，常常会维持较高的采样温度和策略熵。熵越高，下一 token 分布越平，top token 越不确定。此时如果用常见的 target-only / greedy draft 接受方式，接受率天然会被 target 分布的峰值概率限制住。简单说就是：

```text
模型越不确定，drafter 越难猜中那个会被 target 接受的 token。
```

这件事在多步 MTP 里还会被放大。一步接受率掉一点，三步、五步连乘之后，accept length 会明显缩短。rollout 加速就这样被吃掉了。

论文里 target-only sampling 的单步接受率可以近似看成：

$$
\alpha_{\text{TO}} = p(\hat y), \quad \hat y = \arg\max_y q(y)
$$

如果 drafter 能猜中 target 的 top-1 token，那么：

$$
\alpha_{\text{TO}} = \max_y p(y)
$$

而 target 分布的熵是：

$$
H(p) = -\sum_y p(y)\log p(y)
$$

熵越大，$\max_y p(y)$ 往往越小。Bebop 把这个关系写成局部线性近似：

$$
\alpha_{\text{TO}} \approx a_{\text{TO}} - b_{\text{TO}} H(p), \quad b_{\text{TO}} > 0
$$

这就是标题里 `Breaking Entropy Bounds` 的含义：如果接受规则主要吃 top-1 概率，那么 policy 熵一上来，接受率会被直接压住。

## Bebop 的解法：别间接优化 KL，直接优化接受率相关的 TV

Bebop 的方案可以拆成两层。

第一层是把验证方式从 target-only sampling 换成 probabilistic rejection sampling。

在 rejection sampling 里，接受概率由 draft 分布和 target 分布的重叠程度决定。只要按规则处理被拒绝位置，最终输出分布仍然精确等于 target model 的分布。它不是牺牲质量换速度，而是在保持目标分布的前提下提高并行验证收益。

第二层更关键：MTP 的训练目标不要继续只用 CE / KL，而是直接优化 Total Variation distance，也就是 `TV loss`。

为什么？因为 rejection sampling 的接受率，本质上由 draft 分布和 target 分布的 TV distance 决定。CE / KL 当然也在让两个分布接近，但它们是间接目标，优化力会被分配到大量长尾 token 上；而 TV loss 更贴近“这个 token 到底会不会被接受”的判定边界。

传统 CE 可以写成：

$$\mathcal{L}_{\text{CE}} = - \sum_y p(y)\log q(y)$$

KL 则是：

$$D_{\text{KL}}(p\|q) = \sum_y p(y)\log \frac{p(y)}{q(y)}$$

它们都在让 $q$ 靠近 $p$，但它们不是接受率本身。Bebop 要优化的是：

$$\mathcal{L}_{\text{TV}} = d_{\text{TV}}(p, q) = 1 - \sum_y \min(p(y), q(y))$$

因为：

$$\max \alpha_{\text{RS}} \Longleftrightarrow \min d_{\text{TV}}(p, q) \Longleftrightarrow \min \mathcal{L}_{\text{TV}}$$

这比“用 KL 间接约束 TV”更贴近 speculative decoding 的真实目标。

Bebop 进一步提出 `end-to-end TV loss`，不是只平均优化每个 MTP step，而是考虑多步接受的连乘结构。越靠前的位置越重要，因为前面一拒绝，后面的 token 全部没机会被验证。于是训练目标自然会更重视那些决定整段 prefix 能不能活下来的位置。

如果第 $k$ 个 MTP head 的 draft 分布是 $q_k$，target 分布是 $p_k$，那么单步接受概率是：

$$
\alpha_k = 1 - d_{\text{TV}}(p_k, q_k)
$$

多步 prefix 存活概率是：

$$
P(\tau \ge j) = \prod_{k=1}^{j}\left(1 - d_{\text{TV}}(p_k, q_k)\right)
$$

所以 e2e TV loss 的直觉不是“第 1、2、3 步各自像 target 就行”，而是直接优化：

$$\mathbb{E}[\tau] = \sum_{j=1}^{\gamma}\prod_{k=1}^{j}\left(1 - d_{\text{TV}}(p_k, q_k)\right)$$

也就是说，它在训练时就把“前面一拒绝，后面全没了”的 prefix 结构放进目标里。

这套组合带来的方法论很清楚：

```text
不要只让 drafter 学“像不像 target”；
要让 drafter 学“怎样在 rejection sampling 规则下更容易被 target 接受”。
```

这也是我觉得 Bebop 最有价值的地方。它把 speculative decoding 从一个推理 trick，改写成了一个和 RL 采样分布强相关的训练目标问题。

## 为什么 Bebop 不主张在线更新 MTP

一个直接反应是：既然 RL 过程中 policy 变了，那是不是应该边 RL 边更新 MTP？

Bebop 的结论反而偏工程：多数情况下不值得。

论文实验显示，用 `e2e TV loss` 在 RL 前做轻量 MTP adaptation，再在 rollout 中使用 rejection sampling，就足以在整个 RL 过程中维持相对稳定的接受率。相反，在线更新 MTP 会引入额外显存、通信和延迟成本，而且用 CE 继续更新还可能把原本 TV loss 训练出来的分布形状拉坏。

所以 Bebop 的 practical recipe 可以压缩成：

```text
RL 前把 MTP 训到适合 rejection sampling；
RL 中冻结 MTP；
rollout 阶段用 rejection sampling 稳住高熵场景下的 accept length。
```

论文报告的结果也很直接：在 Qwen3.5 / Qwen3.6 / Qwen3.7 的 reasoning、coding、agentic 任务中，TV-trained MTP 的接受率最高可到 95%，并且相对传统 CE/KL 目标带来最多约 25% 的额外推理吞吐收益。对 RL 来说，这种收益不是小修小补，因为 rollout 往往对应成百上千 GPU 小时级别的成本。

## DeepSeek DSpark：线上 serving 的问题不是猜得不够长，而是验证得不够聪明

如果说 Bebop 的敌人是 RL 训练中的高熵 policy，那么 DSpark 的敌人就是线上 serving 里的高并发负载。

在单请求、低负载环境下，speculative decoding 看起来很美：多猜一些 token，target model 一次验证，accepted length 变长，单用户速度上去。

但真实生产环境不是单请求。几十、几百个请求一起进来时，target model 的 batch capacity 是稀缺资源。你多验证一个低置信 token，就可能挤掉另一个请求更有价值的 token。于是长 draft block 未必带来更高吞吐，甚至可能因为验证浪费导致 aggregate throughput 下降。

这就是 DSpark 的核心判断：

```text
draft block 不是越长越好；
真正的问题是每个请求、每个时刻，应该把 target verification budget 花在哪里。
```

## DSpark 的第一块拼图：半自回归 drafter

传统 drafter 大致有两类。

自回归 drafter 会一个 token 一个 token 地猜。好处是每个位置能看到前面已经猜出的 token，局部依赖强；坏处是 draft latency 会随 block size 线性增加。

并行 drafter 一次 forward 猜出一整块 token。好处是快；坏处是块内每个位置相对独立，容易出现后缀接受率快速衰减。比如上下文里有多个合理续写模式，并行预测可能把不同模式的片段混在一起，前几个 token 还行，后面的组合就开始不自然。

DSpark 的做法是折中：**主体仍然是 heavy parallel backbone，但在输出端加一个 lightweight sequential module**。论文里主要使用 Markov head，也讨论了 RNN head 变体。

用公式写，它不是让整个 draft block 完全独立：

$$
P(X \mid x_0) \ne \prod_{k=1}^{\gamma} p_k(x_k \mid x_0)
$$

而是让 block 内部仍然按一个轻量的因果分解走：

$$P(X \mid x_0) = \prod_{k=1}^{\gamma}p_k(x_k \mid x_0, x_{<k})$$

其中每个位置的分布由 parallel backbone 的 base logit $U_k$ 加上 sequential head 的 transition bias $B_k$ 得到：

$$p_k(v \mid x_0, x_{<k}) = \frac{\exp(U_k(v)+B_k(x_0,x_{<k},v))}{\sum_{u \in V}\exp(U_k(u)+B_k(x_0,x_{<k},u))}$$

Markov head 是最轻的实现，它只看前一个 token：

$$
B(x_{k-1}, \cdot) = W_1[x_{k-1}]W_2
$$

这里的意思很直白：parallel backbone 负责一次性给出每个位置的大致方向，Markov/RNN head 负责补上“前一个词已经选了这个，那下一个词应该更像哪个模式”的局部约束。

这个设计的含义是：

```text
大部分计算保持并行；
只用很轻的一层顺序结构，把 draft token 之间的局部转移关系补回来。
```

它不是彻底回到自回归，而是在并行生成的基础上补一点必要的因果依赖。这个“一点点自回归”很关键：它缓解了 suffix decay，同时没有把 draft latency 拉回线性增长。

## DSpark 的第二块拼图：置信度调度验证长度

有了更强的 drafter，仍然不能无脑验证整段 draft。DSpark 的另一半是 `confidence-scheduled verification`。

它给每个 draft position 预测一个条件概率：在前面 token 都被接受的情况下，当前位置 token 能不能继续活下来。把这些条件概率连乘，就得到某个 prefix 整体存活的概率。

具体地，confidence head 输出：

$$
c_k \in (0,1)
$$

它表示：

$$c_k \approx P(x_k \text{ survives} \mid x_1,\dots,x_{k-1}\text{ all survive})$$

DSpark 用 TV 距离给这个 confidence 一个软标签：

$$c_k^{\ast} = 1 - \frac{1}{2}\|p_k^d - p_k^t\|_1 = 1 - d_{\text{TV}}(p_k^d, p_k^t)$$

于是 prefix 到第 $j$ 个 token 仍然存活的概率就是：

$$
a_j = P(\tau \ge j) = \prod_{i=1}^{j} c_i
$$

然后 scheduler 不再用固定长度验证，而是根据两类信息做选择：

- 当前请求的 prefix survival probability；
- 当前 serving engine 在不同 batch size 下的吞吐曲线。

直观理解就是：

```text
低负载时，多验证一些 token，因为闲着的 target compute 不用白不用；
高负载时，砍掉低置信 suffix，因为它们会占用稀缺 batch capacity。
```

这一步把 speculative decoding 从“模型推理算法”推到了“系统调度算法”。尤其在高并发场景下，验证长度不再只是模型置信度问题，而是全局 throughput maximization 问题。

假设当前有 $R$ 个请求。第 $r$ 个请求选择验证长度 $\ell_r$。那么送进 target model 的 token batch size 是：

$$
B = \sum_{r=1}^{R}(1+\ell_r)
$$

这里的 `1` 是每个请求至少需要 target model 产生的 bonus / anchor token。期望成功产出的 token 数可以写成：

$$\tau = \sum_{r=1}^{R}\left(1 + \sum_{j=1}^{\ell_r} a_{r,j}\right)$$

如果 `SPS(B)` 表示 serving engine 在 batch size 为 $B$ 时每秒能跑多少个 decode step，那么系统期望吞吐是：

$$\Theta(\ell_1,\dots,\ell_R) = \tau \cdot \text{SPS}(B)$$

DSpark 的 scheduler 本质上就是解：

$$\ell_1^{\ast}, \dots, \ell_R^{\ast} = \arg\max_{\ell_1,\dots,\ell_R}\Theta(\ell_1,\dots,\ell_R)$$

这条式子解释了为什么“固定验证 16 个 token”不是最优：多验证一个 token 的收益是它的 prefix survival probability $a_{r,j}$，但代价是把 $B$ 变大，可能让 `SPS(B)` 掉到硬件吞吐曲线的下一个台阶。

DSpark 还做了一个容易被忽略但很重要的细节：置信度需要校准。原始 confidence head 往往过度自信，直接拿来估算 prefix survival 会让调度器误判收益。论文使用 `Sequential Temperature Scaling` 做后验校准，把分数调成更接近真实接受率的概率。

## DSpark 真正厉害的地方：它为生产系统做了让步

很多论文方法在离线 benchmark 上成立，但一进生产 serving 就会碰到 CUDA graph、zero-overhead scheduling、batch shape、KV cache、variable-length query 等一堆细节。

DSpark 有意思的地方，是它没有把这些当成附录里的“小工程问题”。

比如硬件吞吐曲线不是平滑的，而是有很多台阶和 cliff；理论 scheduler 的 early stopping 在这种离散曲线上未必最优。再比如动态验证长度会破坏固定 query length 的高效 kernel 形态。DeepSeek 的处理方式包括：

- 用异步调度避免 GPU pipeline 被同步 scheduler 卡住；
- 用历史两步的 confidence 预测近似下一步 capacity，让调度开销被隐藏；
- 把不同请求的 variable-length verification tokens 展平处理，再用 marker tensor 传递序列依赖；
- 在 DeepSeek-V4 架构里只改必要的 index-attention 和 compress kernels。

这些东西读起来不如“新 loss”漂亮，但它们决定了方法能不能真的在生产里跑。

论文在 DeepSeek-V4-Flash 和 DeepSeek-V4-Pro 的 live traffic 上报告了结果：在中等 SLA 下，DSpark 相比 MTP-1 baseline 提升约 51% / 52% 的 aggregate throughput；在 matched throughput 下，V4-Flash 的 per-user generation speed 提升 60% 到 85%，V4-Pro 提升 57% 到 78%。更重要的是，在更严格的 interactivity SLA 下，baseline 会掉进低并发低利用率区间，而 DSpark 还能维持可用吞吐，相当于把 serving Pareto frontier 往外推了一截。

## 两篇论文放在一起，差异在哪里

我会把它们的差异整理成这样：

| 维度 | Qwen Bebop | DeepSeek DSpark |
| :--- | :--- | :--- |
| 主要场景 | RL 后训练 rollout | 线上 LLM serving |
| 核心瓶颈 | 高熵 policy 让 MTP 接受率下降 | 高并发下验证低置信 token 浪费 batch capacity |
| 关键对象 | MTP 训练目标和接受方式 | drafter 架构和验证调度 |
| 主要解法 | rejection sampling + e2e TV loss + pre-RL adaptation | semi-autoregressive drafter + confidence head + hardware-aware scheduler |
| 系统目标 | 缩短 RL wall-clock time | 同时提高单用户速度和系统吞吐 |
| 方法论 | 让 draft 分布更适合被 target 接受 | 把 target verification budget 花在最值的位置 |

一个偏训练分布，一个偏服务调度；一个把 loss 对齐到 acceptance rule，一个把 verification length 对齐到 engine load。

但它们的共同点也很强：

**接受率不是一个附属指标，而是整套系统的主优化对象。**

在 Bebop 里，接受率连接的是 `policy entropy -> draft-target overlap -> rollout latency -> RL 训练成本`。

在 DSpark 里，接受率连接的是 `draft confidence -> verification budget -> batch capacity -> serving Pareto frontier`。

同一个 accept length，到了训练系统里是 GPU 小时，到了 serving 系统里是用户感知延迟和并发吞吐。

## 我觉得最重要的 insight

如果把两篇论文共同的 insight 压成一句话，我会写成：

**speculative decoding 不是“用小模型猜大模型”，而是在目标分布不变的约束下，最大化每一次 target forward 的有效产出。**

这句话里有三个关键词。

第一个是“目标分布不变”。无论 Bebop 还是 DSpark，都很重视 rejection sampling 的 exactness。加速不能靠偷偷改输出分布，否则它就不再是等价加速，而是另一个模型。

第二个是“有效产出”。不是 draft 越多越好，而是 accepted tokens 越多越好；不是验证越多越好，而是单位 target compute 换来的 accepted tokens 越多越好。

第三个是“系统”。Bebop 说明训练时的 entropy、temperature、RL rollout 形态会改变 speculative decoding 的收益；DSpark 说明线上 serving 的 concurrency、kernel、batch capacity 也会改变收益。算法如果不看系统，收益很容易只停在图表里。

## 对之后做 LLM 系统的启发

我觉得这两篇论文对后面的 LLM 工程有几个很实用的启发。

第一，推理加速方法不能只在静态 benchmark 上测平均速度。必须问它在高熵采样、长生成、多轮工具调用、高并发 serving 下是否仍然稳定。

第二，draft model 的训练目标应该贴近最终使用的验证规则。用 CE 训练一个“看起来像 target”的 drafter，不一定等于训练了一个“容易被 target 接受”的 drafter。

第三，线上 speculative decoding 应该是 load-aware 的。固定 draft length / verify length 在单请求下简单，但在生产环境里很容易浪费最贵的 target compute。

第四，训练系统和 serving 系统会越来越共享同一批底层思想。RL rollout 本质上也是大规模推理；serving speculative decoding 里的分布匹配、接受率、调度问题，也会反过来影响训练吞吐。

## 结语

如果只把 Bebop 和 DSpark 看成“又快了多少百分比”，会错过更有意思的变化。

Bebop 说的是：在 RL 这种分布不断变化的训练场景里，speculation 要和 entropy、sampling rule、loss function 对齐。

DSpark 说的是：在真实 serving 这种负载不断变化的生产场景里，speculation 要和 confidence、batch capacity、hardware scheduler 对齐。

它们合在一起，其实给了 speculative decoding 一个更成熟的定义：

```text
不是提前生成 token，
而是在保持 target 分布精确不变的前提下，
把每一次大模型 forward 尽可能变成更多有效 token。
```

从这个角度看，speculative decoding 的下一阶段不会只是更小的 drafter 或更长的 block，而会是更完整的系统闭环：分布怎么训、token 怎么验、预算怎么分、负载怎么调，以及这些决策怎样在训练和线上同时成立。

## 参考链接

- [Breaking Entropy Bounds: Accelerating RL Training via MTP with Rejection Sampling - arXiv](https://arxiv.org/abs/2606.12370)
- [Breaking Entropy Bounds PDF](https://arxiv.org/pdf/2606.12370)
- [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation - arXiv](https://arxiv.org/abs/2607.05147)
- [DSpark PDF](https://arxiv.org/pdf/2607.05147)
- [DeepSeek-V4-Pro-DSpark on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark)
- [DeepSpec GitHub](https://github.com/deepseek-ai/DeepSpec)

