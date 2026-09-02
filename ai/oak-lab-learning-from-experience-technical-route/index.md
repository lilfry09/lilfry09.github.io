# Oak Lab：从经验学习到实时智能的技术路线


Oak Lab 最近的一篇文章，标题是 [Learning from experience instead of curated datasets](https://oaklab.ai/posts/learning-from-experience-instead-of-curated-datasets)。它表面上在比较 SGD 与 NetworkIDBD，实际上是在提出一个更大的研究议程：**如果智能体面对的是未经筛选、持续到来的真实经验，学习算法本身必须知道哪些误差值得学习，哪些误差只是噪声。**

这与“把更多数据喂给更大的 Transformer”是不同的问题。Oak Lab 的公开使命是让智能体在“大世界”（big worlds）中实现目标：环境持续变化、观测不完整、资源有限，系统不能把所有知识都同时精确地存下来。围绕这个目标，他们把研究集中到 OaK Architecture、batch-size-one 在线学习、事件驱动神经网络和面向规划的时间抽象上。

本文信息截止 **2026-09-02**。Oak Lab 尚未公开 OaK 的完整实现、训练数据配方或端到端能耗测量，因此文中会明确区分来源结论与综合判断。

<!--more-->

## 30 秒结论

1. **问题不是“数据有没有噪声”，而是学习规则是否会把噪声也归因给参数。** 在人工整理的数据集里，数据通常已经被假定为有用；在经验流中，这个假设不成立。
2. **Oak Lab 的核心技术押注是可学习的 credit assignment。** IDBD/NetworkIDBD 通过为不同参数学习不同步长，尝试只强化可预测的关联，而不是让所有非零梯度参数共同承担误差。
3. **算法形态是 online、continual、batch-size one。** 目标是在不存储或 replay 历史数据的情况下，直接从每一步经验更新。
4. **硬件与算法被一起设计。** 事件驱动神经网络只在事件发生时计算，Oak Lab 认为这能把在线学习的计算和能耗降到传统 dense、mini-batch 系统难以达到的水平。
5. **OaK Architecture 是上层目标。** 它要从经验中发现时间抽象，这些抽象既能自我验证，又能用于规划；但公开资料还不足以复现其具体模块。

## 1. 文章到底证明了什么？

### 1.1 先构造一个“经验流”

Oak Lab 先给出一个极简监督预测流。每个时间步只有一个二值特征 $x_t$，以 1% 的概率取 1；目标 $y_t$ 与该特征相同。这个任务本身很容易：线性模型只要学到一个接近 1 的权重即可。

随后他们逐步加入现实经验中的两个因素：

- **不可预测的目标噪声**：以 1% 的概率加入随机的 $+1$ 或 $-1$，再加入均值为 0、方差为 5 的高斯噪声；
- **无关输入**：再加入 4095 个同样稀疏、但与目标不相关的 Bernoulli 特征。

于是，数据流中同时存在“可学习关联”和“不可学习扰动”。这正是人工清洗数据集通常会提前消除的部分。

### 1.2 为什么 SGD 会学坏？

对线性预测器，普通 SGD 的更新可以写成：

$$
w_{t+1}=w_t-\alpha\nabla_w\ell_t(w_t),
$$

其中 $\alpha$ 是统一步长，$\ell_t$ 是当前样本的损失。只要某个特征在当前样本中激活，它就可能通过梯度获得更新；算法没有内置机制判断“这个误差来自可泛化信号，还是来自不可预测噪声”。

Oak Lab 的实验观察是：在 4096 个输入特征的设置中，SGD 会把噪声吸收到权重里，而不是只学习第一个、真正与目标相关的特征。Adam、RMSProp 等常用变体仍以最小化所有样本误差为目标，不能从根本上解决这个 credit assignment 问题。

### 1.3 IDBD 的不同之处

IDBD（Incremental Delta-Bar-Delta）不为所有参数使用同一个固定步长，而是让步长本身随数据流学习。可以把它抽象成：

$$
w_{t+1}=w_t+\alpha_t\odot g_t,
$$

其中 $g_t$ 是当前梯度，$\alpha_t$ 是按参数区分的、可在线调整的步长。关键不是公式形式本身，而是**信用分配从“误差发生了，所以所有相关参数都更新”变成“长期看，哪些信号值得继续更新？”**

Oak Lab 在同一任务、同样步数下比较 IDBD 与 SGD，报告 IDBD 能保留可预测部分，同时抑制不可预测目标带来的更新。

## 2. NetworkIDBD：把选择性信用分配带进神经网络

线性例子还不够，因为真实感知任务需要先学习非线性特征。Oak Lab 构造了 NoisyMNIST 数据流：

- 将 28×28 的 MNIST 数字放到 64×64 输入的中心区域；
- 10% 的时间步出现数字，奇数目标为 $+1$，偶数目标为 $-1$，没有数字时目标为 0；
- 其余像素是与目标无关的稀疏噪声特征；
- 目标继续加入方差为 5 的高斯噪声。

网络包含一个 10,000 单元的 ReLU 隐藏层。训练后，Oak Lab 观察权重幅度超过阈值的连接：SGD 会增强来自所有输入特征的连接；NetworkIDBD 则主要增强中心 28×28 数字区域的连接。

【来源结论】这个可视化支持这样的判断：NetworkIDBD 在该实验中更倾向于把信用分配给稳定、可预测的输入区域。

【证据边界】这不是“NetworkIDBD 已经解决了真实世界在线学习”的证明。实验是受控的监督流，模型规模、任务和噪声分布都很小；它证明的是一种机制上的可行性。

## 3. Oak Lab 的技术路线：从底层更新规则到 OaK

把 Oak Lab 官网的 mission、研究列表和上述文章放在一起，可以得到下面这条链：

```mermaid
flowchart LR
    A[连续经验流] --> B[选择性 credit assignment]
    B --> C[Batch-size-one 在线更新]
    C --> D[不存储/不 replay 的持续学习]
    D --> E[事件驱动神经网络]
    E --> F[低计算与低能耗的实时智能体]
    F --> G[时间抽象]
    G --> H[自我验证的规划]
    H --> I[OaK Architecture]
```

### 3.1 第一层：学习什么——可泛化关联，而不是所有相关性

传统优化通常把每个样本误差都当成训练信号。Oak Lab 的路线要求算法估计某个参数更新的长期价值：如果一个输入只与偶然噪声相关，步长应逐渐减小；如果一个输入在时间上稳定地预测目标，步长应保留甚至增大。

这解释了他们为什么持续研究 IDBD、NetworkIDBD、step-size optimization 和 SwiftTD：这些工作共同指向“让更新规则自己学习如何更新”。

### 3.2 第二层：如何学习——batch-size one 与实时更新

Oak Lab mission 明确写道，他们的算法可以直接从经验学习，不需要存储或 replay 数据。这里的 batch-size one 不是把大 batch 简单设成 1，而是改变系统假设：

- 数据按时间顺序到达；
- 每个样本只被处理一次；
- 更新必须足够便宜，能跟上环境；
- 算法要利用时间相邻样本之间的 temporal coherence。

这条路线天然接近 reinforcement learning 和 continual learning，而不是离线 supervised pretraining。

### 3.3 第三层：如何节省算力——事件驱动计算

Oak Lab 公开表示，batch-size-one 学习算法与 event-driven neural networks 结合后，可以比现有方法少用多个数量级的计算和能量。事件驱动网络的基本思想是：神经元状态只有在输入或内部状态发生足够变化时才触发计算，而不是每个时间步对所有单元做 dense 更新。

【综合判断】这是一种“算法—硬件协同”路线：选择性信用分配减少无效参数更新，事件驱动执行减少无效前向计算，两者共同服务于实时、低功耗学习。官网没有公开统一的 benchmark、芯片型号或端到端测量脚本，因此“多个数量级”目前应视为 Oak Lab 的公开目标/主张，而非可独立复核的通用结论。

### 3.4 第四层：学到什么结构——时间抽象

OaK Architecture 的公开描述是：从经验中发现 temporal abstractions，并要求这些抽象同时满足两点：

1. **self-verifiable**：智能体能够检验抽象是否仍然预测经验中的变化；
2. **useful for planning**：抽象不是只用于压缩表示，还能作为规划和决策的中间结构。

这比“学一个更好的 embedding”更强。它暗示模型需要发现事件、状态转移、子目标或可重复的时间片段，并让这些结构进入 planning loop。

但目前公开页面没有给出 OaK 的层数、状态变量、损失函数或规划器接口。因此更稳妥的表述是：OaK 是 Oak Lab 的架构目标和研究方向，而不是已经完全公开的可复现模型。

## 4. 把 NetworkIDBD 拆成可执行的在线算法

为了理解 Oak Lab 的技术选择，可以把 NetworkIDBD 看成三条同时运行的状态更新：预测状态、参数状态和步长状态。下面不是 Oak Lab 公布的完整伪代码，而是根据 IDBD 家族的公开定义抽象出的最小执行图。

设网络在时间步 $t$ 接收输入 $x_t$，输出 $hat y_t=f_{w_t}(x_t)$，即时损失为

$$
\ell_t=\frac{1}{2}(y_t-\hat y_t)^2.
$$

普通 SGD 只维护参数 $w_t$：

$$
g_t=\nabla_{w}\ell_t,\qquad
w_{t+1}=w_t-\alpha g_t.
$$

IDBD 还为每个参数 $i$ 维护一个 log step-size $eta_{i,t}$，令

$$
\alpha_{i,t}=\exp(\beta_{i,t}).
$$

于是参数更新变成逐参数的形式：

$$
w_{i,t+1}=w_{i,t}-\exp(\beta_{i,t})g_{i,t}.
$$

关键在于 $eta_{i,t}$ 也会根据“改变该参数的步长是否让未来损失下降”而更新。为此算法维护一个与参数同形状的 eligibility / sensitivity 状态 $h_{i,t}$，用它近似当前参数对历史预测的影响。不同论文对 $h_{i,t}$ 的具体递推和符号约定略有差异，但抽象结构可以写成：

$$
h_{i,t}=\rho_t h_{i,t-1}+\frac{\partial \hat y_t}{\partial w_{i,t}},
$$

$$
\beta_{i,t+1}=\beta_{i,t}-\eta\,\frac{\partial \ell_t}{\partial \beta_{i,t}},
$$

其中 $ho_t$ 控制历史影响的衰减，$eta$ 是 meta step-size。直觉是：若某个输入通道的更新长期能降低可预测误差，$eta_i$ 会保持较大；若它只响应随机噪声，继续增大步长不会带来稳定收益，$eta_i$ 会下降。

一个 batch-size-one 的最小循环如下：

```text
初始化 w、β、h
for each experience (x_t, y_t):
    y_hat = f_w(x_t)
    loss  = 1/2 * (y_t - y_hat)^2
    g     = gradient(loss, w)
    h     = update_sensitivity(h, w, x_t)
    β     = update_step_size(β, loss, g, h)
    w     = w - exp(β) ⊙ g
```

这个循环体现了 Oak Lab 关注的三个工程约束：每个样本只来一次；状态 $w,\beta,h$ 随时间流动；更新量必须足够小，不能依赖事后 replay 全部历史。

### 4.1 为什么“选择性”不是简单的梯度裁剪？

梯度裁剪通常处理的是数值稳定性：当 $\lVert g_t\rVert$ 过大时缩放整个梯度。它不区分“有用的大梯度”和“无用的大梯度”。IDBD 类方法处理的是统计信用：某个参数是否应该在未来继续对类似信号保持敏感。

二者可以同时使用，但含义不同：

| 机制 | 解决的问题 | 时间尺度 |
| --- | --- | --- |
| gradient clipping | 防止单步更新爆炸 | 单个样本 |
| momentum / Adam moments | 平滑近期梯度 | 短期窗口 |
| IDBD step-size state | 学习每个参数的长期更新价值 | 跨样本、持续运行 |

这也是 Oak Lab 认为 Adam、RMSProp 仍不够的原因：它们改进了梯度的尺度和方向估计，却没有显式学习“这个误差是否值得归因给这个信号”。

### 4.2 从线性特征到深层表征的困难

在线更新一旦进入深层网络，信用分配就不再只对应输入特征。一个隐藏单元可能同时接收可预测像素和噪声像素；如果只对输出层使用自适应步长，前层仍可能把噪声编码进表示。因此 NetworkIDBD 的实验重点不只是最终预测误差，还包括输入到隐藏层连接的可视化：它试图回答“噪声究竟有没有进入表示”。

这带来一个可检验的扩展问题：在更深网络中，应该为每一层、每个通道，还是每个权重维护步长状态？状态粒度越细，选择性越强，但额外内存也越大。若参数量为 $P$，同时维护 $\beta$ 和 $h$，朴素实现的优化状态可能从 $P$ 增加到约 $3P$；这会直接影响 Oak Lab 所追求的低能耗目标。

## 5. NoisyMNIST 实验的完整因果链

NoisyMNIST 的价值在于它把“可预测结构”和“不可预测结构”放进同一个流中，同时保留了视觉任务需要的非线性特征学习。可以按生成顺序重建：

1. 以 10% 概率采样一张 MNIST 数字，否则中心区域为空；
2. 将数字放到 64×64 画布的中心 28×28 区域；
3. 若出现数字，奇数给目标 $+1$，偶数给目标 $-1$；若没有数字，目标为 0；
4. 其余像素作为独立 Bernoulli 噪声，每个位置以 1% 概率激活；
5. 对目标加入 $\epsilon_t\sim\mathcal N(0,5)$，得到最终观测目标 $y_t$。

因此可以写成：

$$
y_t = z_t\,s(d_t)+\epsilon_t,
$$

其中 $z_t\in\{0,1\}$ 表示是否出现数字，$d_t$ 是数字类别，$s(d_t)=+1$（奇数）或 $-1$（偶数）；当 $z_t=0$ 时令 $s(d_t)=0$。中心区域与 $y_t$ 存在稳定关联，外围噪声像素的期望相关性为 0。

训练后比较的不是“哪一个模型偶然得到更低 loss”，而是参数空间中的选择性：

- SGD 若把所有输入连接都增强，说明噪声被写入隐藏表示；
- NetworkIDBD 若主要增强中心区域连接，说明更新规则利用了跨样本稳定性。

这是一种很好的机制实验设计，因为它同时提供行为指标（预测）和结构指标（权重归因）。但它仍缺少长期非平稳性：当前任务的可预测区域不会随时间迁移，而真实经验中有用信号可能出现、消失、再出现。

## 6. Big World 下的资源账本：不只是参数量

Oak Lab 所说的“资源受限”至少包含四种不同预算，不能用参数量一个数字代替：

| 资源 | 典型瓶颈 | 对在线智能体的含义 |
| --- | --- | --- |
| compute | 每步可执行的算子数量 | 决定能否跟上环境频率 |
| memory capacity | 可保留的参数、状态和缓存 | 决定能记住多少经验 |
| memory bandwidth | 状态读写速度 | 决定稀疏更新是否真的更快 |
| energy | 传感、计算、通信总功耗 | 决定能否长期部署 |

在大世界中，增加模型参数可能提高表示能力，却也增加每步读写和更新成本。Oak Lab 因此强调 approximate but efficient algorithms：如果把省下的计算用于扩大表示或提高交互频率，总体决策质量可能优于更精确但更昂贵的算法。

一个简单的决策目标可以写成质量—资源联合优化：

$$
\max_\pi\; J(\pi)-\lambda_C C(\pi)-\lambda_E E(\pi),
$$

其中 $J(\pi)$ 是策略长期收益，$C(\pi)$ 是计算成本，$E(\pi)$ 是能耗，$\lambda_C,\lambda_E$ 将资源约束映射到同一目标。Oak Lab 的公开文章没有给出他们实际使用的联合目标；这个式子只是解释为什么 benchmark 必须同时报告质量、延迟和能耗。

## 7. 事件驱动神经网络：算法状态如何映射到硬件执行

传统 dense 网络在每个时间步对所有单元执行矩阵乘法。事件驱动网络则维护神经元状态 $u_t$，只有当输入变化或膜电位超过阈值时才产生事件 $s_t$。一个简化的 leaky integrate-and-fire 更新是：

$$
u_t=\lambda u_{t-1}+W s_t,
$$

$$
s_t=\mathbf 1[u_t\ge\vartheta],
$$

其中 $\lambda$ 是泄漏系数，$\vartheta$ 是触发阈值。若 $s_t$ 大部分时间为 0，系统可以跳过大量不必要的乘加。

但“事件稀疏”不自动等于“端到端更省”：

1. 稀疏索引和事件队列会引入调度开销；
2. 权重读取可能比乘法本身更耗能；
3. batch-size one 降低了矩阵乘法的并行度；
4. 反向传播或 meta-gradient 可能需要额外状态存储。

所以 Oak Lab 的路线必须同时优化学习规则、事件表示和硬件映射。真正有说服力的报告应给出 event rate、每事件算子数、内存访问量、平均延迟和任务质量，而不只是理论 FLOPs。

## 8. OaK Architecture：从预测器到可规划的时间抽象

根据 Oak Lab 的 mission，OaK 的目标不是一个单独的分类器，而是一个能从经验中发现时间抽象、验证抽象并用于规划的 agent architecture。用功能模块表示，可以拆成四个接口：

```text
experience stream
      ↓
perception / state estimator
      ↓
temporal abstraction learner ──→ self-verification signal
      ↓
predictive model / value estimates
      ↓
planner ──→ action
      ↑             │
      └── new experience ──┘
```

### 8.1 时间抽象至少要回答三个问题

**边界在哪里？** 抽象可能从事件触发、状态变化率或预测误差突变中发现起点和终点。

**抽象预测什么？** 它可以预测下一个宏观状态、持续时间、奖励，或一组可达状态分布。

**为什么能用于规划？** 规划器需要把抽象当作 action / option / transition 使用，并能估算组合多个抽象后的结果。

一种通用的抽象表示可以写成 $o=(\phi,\tau,\psi)$：$\phi$ 是启动条件，$\tau$ 是持续或终止条件，$\psi$ 是对结果状态的预测。若抽象是可验证的，则系统应能在新经验到达后计算预测误差：

$$
e_t^{(o)}=d\big(\psi_o(s_t),s_{t+\Delta t}\big),
$$

并据此调整抽象的可信度或适用范围。需要强调：Oak Lab 尚未公开 OaK 是否采用这种 option 形式；它只是把“self-verifiable and useful for planning”翻译成可实现接口的一种候选方案。

### 8.2 为什么 OaK 需要在线学习底座？

时间抽象不是一次性从静态数据集聚类出来就结束。环境变化后，抽象的持续时间、结果分布和适用条件都会变化；如果没有 tracking 和 continual update，规划器会继续使用过时模型。于是 OaK 的上层规划能力依赖底层更新规则能否：

- 快速吸收新规律；
- 不因短期噪声而重写长期规律；
- 在资源不足时淘汰低价值状态；
- 对抽象的预测失败提供可追踪的信用信号。

这正是“从经验学习”与“时间抽象”之间的接口，而不是两个并列口号。

## 9. 一套更严格的验证协议

如果要把 Oak Lab 的主张推进到可复现实验，我建议至少报告以下四条曲线：

1. **质量—样本数**：在每个样本只出现一次的前提下，累计预测误差或任务回报；
2. **质量—计算**：横轴使用实际 executed operations 或 wall-clock，而不是只报参数量；
3. **遗忘—分布变化**：在可预测区域迁移、噪声强度变化、目标反转时测旧任务保持率；
4. **稀疏率—能耗**：报告事件率、状态读写和端到端功耗，而非仅根据理论稀疏度估算。

最小对照应包括：SGD、Adam/RMSProp、带 replay 的 continual baseline、IDBD/NetworkIDBD，以及同等每步计算预算下的 dense 与 event-driven 实现。若只比较最终准确率，无法判断收益来自更好的 credit assignment、更多计算，还是更强的记忆。

## 10. Big World Hypothesis：为什么他们不满足于离线大模型

Oak Lab 的另一篇公开文章把研究动机概括为 **Big World Hypothesis**：在许多决策问题中，环境的复杂度远大于单个智能体的感知、记忆和计算能力。智能体不可能同时精确表示所有状态、价值和最优动作，只能学习对当前目标足够好的近似。

这个假设带来三个直接后果：

| 后果 | 对算法的要求 |
| --- | --- |
| 世界持续变化 | online continual learning，而非一次性训练后冻结 |
| 资源有限 | 计算高效的近似算法，必要时牺牲全局精确性 |
| 重要性随时间变化 | tracking：学习当前有用的知识，淘汰暂时无用的知识 |

因此，Oak Lab 的目标不是建立一个“记住一切”的静态模型，而是让一个受限智能体在不断变化的环境中持续追踪有用结构。这也是他们把 temporal coherence、continual learning 和资源约束放在一起讨论的原因。

## 11. 研究谱系：官网列出的项目如何相互衔接

Oak Lab 官网列出的 prior research 并非随机论文清单，可以按技术问题分成四组：

1. **增量信用分配与步长学习**：IDBD、Step-size Optimization、SwiftTD。解决“单样本更新应该信任谁、信任多少”。
2. **实时递归与持续学习**：Horde、Columnar-Constructive Networks、Continual Learning。解决“如何在长期数据流中保持实时更新和有限资源”。
3. **大世界中的近似决策**：Big World Hypothesis、Reward-Respecting Subtasks。解决“环境比智能体大很多时，如何用有限表示完成目标”。
4. **架构层整合**：OaK Architecture。目标是把经验学习、时间抽象和规划组合成完整智能体。

这条谱系的关键转折点是：**从改进某个 optimizer，走向重新定义 agent 的学习闭环。** optimizer 只决定参数如何更新；OaK 试图进一步决定什么经验值得保留、哪些抽象可以用于预测，以及规划如何调用这些抽象。

## 12. 与当前主流 LLM 路线的差别

| 维度 | curated-dataset LLM | Oak Lab 路线 |
| --- | --- | --- |
| 数据 | 预先收集、清洗、去重、混合 | 连续、噪声大、不可完全预测的经验流 |
| 更新 | 大 batch、可多轮 replay | batch-size one、在线、尽量一次处理 |
| 信用分配 | 梯度对所有相关参数传播 | 学习参数级或信号级更新信用 |
| 记忆 | 依赖参数、数据集和 replay buffer | tracking：按当前目标保留和替换知识 |
| 计算 | dense、同步、固定频率 | event-driven、按事件触发 |
| 推理结构 | 预测下一个 token 为主 | 预测、时间抽象与规划闭环 |

这不意味着 Oak Lab 的路线已经替代 Transformer。更准确的说法是：他们在研究一套适合“持续交互智能体”的学习基础设施，而当前 LLM 更擅长“从整理好的大规模语料中离线获得通用表示”。两者未来可能结合，但接口尚未公开。

## 13. 哪些地方最值得怀疑？

### 13.1 选择性信用分配是否能扩展到深度网络？

NoisyMNIST 说明机制在小型网络上有效，但深度网络存在 credit assignment 的时间跨度、归一化、优化稳定性和灾难性遗忘问题。需要公开更大规模、跨任务、长时间流的实验。

### 13.2 不 replay 是否真的可行？

不存储历史数据能降低内存和隐私成本，但也放弃了常见的离线纠错手段。若环境分布突然变化，系统如何区分“新规律”与“短期噪声”，仍需要更强的稳定性证据。

### 13.3 低能耗主张如何验证？

比较能耗不能只看 FLOPs。至少应同时报告硬件、批大小、事件稀疏率、内存访问、端到端延迟和任务质量。Oak Lab 公开页面目前没有给出完整测量条件。

### 13.4 OaK 的“自我验证”具体是什么？

可能的实现包括预测误差、模型不确定性、可逆状态转移或基于规划结果的反事实检验，但这些只是候选机制。没有论文或代码前，不应把其中任何一种写成 OaK 已采用的设计。

## 14. 我对这条路线的判断

【综合判断】Oak Lab 最有辨识度的地方，不是单独提出了某个新网络，而是把四个通常分开研究的问题连成一条因果链：

> 经验含噪 → 需要选择性 credit assignment → 才能做 batch-size-one 在线学习 → 才值得用事件驱动硬件追求低能耗 → 最终支撑持续的时间抽象与规划。

这条路线的成败取决于中间环节能否真正闭环。若 NetworkIDBD 只能在受控任务上过滤噪声，无法在复杂视觉或交互环境中稳定追踪；若事件驱动网络节省了算力，却牺牲了表示质量；若 OaK 的抽象不能被规划器可靠调用，那么“从经验到超级智能”的链条就会在某一层断开。

最值得关注的下一步不是宣传中的参数规模，而是三类可复现实验：

- 在长时间、非平稳、单样本数据流上，与 replay-based baseline 比较质量—计算曲线；
- 报告选择性更新对噪声特征、灾难性遗忘和分布突变的影响；
- 给出 OaK 抽象被规划器调用后带来的可测决策收益，并公开完整能耗测量条件。

## 一句话带走

> Oak Lab 想解决的不是“如何把 curated dataset 做得更大”，而是“当世界本身就是噪声、变化和不可预测时，智能体如何只学习值得学习的东西，并用有限能量持续行动”。

## 参考资料

- [Learning from experience instead of curated datasets](https://oaklab.ai/posts/learning-from-experience-instead-of-curated-datasets)，Oak Lab，2026-07-13。
- [Oak Lab Mission](https://oaklab.ai/mission.html)，访问于 2026-09-02。
- [The Big World Hypothesis and its Ramifications for Artificial Intelligence](https://oaklab.ai/posts/the-big-world-hypothesis.html)，Khurram Javed、Richard S. Sutton。
- [The OaK Architecture: A Vision of SuperIntelligence from Experience](https://oaklab.ai/posts/the-oak-architecture.html)，Richard S. Sutton，2025。
- Sutton, R. S. (1992). *Adapting Bias by Gradient Descent: An Incremental Version of Delta-Bar-Delta*.
- [SwiftTD: A Fast and Robust Algorithm for Temporal Difference Learning](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_111.pdf)。
- [Step-size Optimization for Continual Learning](https://arxiv.org/abs/2401.17401)。
- [Scalable Real-Time Recurrent Learning Using Columnar-Constructive Networks](https://www.jmlr.org/papers/volume24/23-0367/23-0367.pdf)。

