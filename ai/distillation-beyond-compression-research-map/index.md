# 蒸馏不只是压缩：从 Self-Distillation 到 Search Distillation 的研究地图


最近我在整理蒸馏方向时，最大的感觉是：这个词被说窄了。

很多人一提 `distillation`，脑子里先出现的是模型压缩、logit 对齐、teacher/student 这一套老图景。但一旦把它放回后训练和推理系统里看，蒸馏真正传递的常常不是“答案”，而是搜索方式、更新方式、判断方式，甚至是数据该怎么长出来。

如果把它重新定义成“一个学习过程，把可迁移的信息结构交给另一个学习过程”，那它就不再只是压缩术，而是一张更大的信息迁移图。

<!--more-->

## 先换一个定义

我更愿意把蒸馏理解成下面这件事：

```text
Model → Model
Model → Data
Model → Weight
Model → Optimizer
Model → Verifier
Search → Model
RL → SFT
SFT → RL
Inference → Training
Teacher → Teacher
Student → Student
Model A ↔ Model B
```

这张图的重点不在“谁教谁”，而在“信息到底从哪条路径迁移”。只要你接受这个定义，很多原本看起来彼此无关的工作就会连起来。

## 一张研究地图

| 方向 | 核心问题 | 为什么重要 |
| --- | --- | --- |
| Self-Distillation | 能不能自己教自己 | 同一模型在不同信息状态下学习 |
| On-policy Distillation | 能不能在 student 自己会走到的状态上学习 | 把稀疏奖励变成更密的反馈 |
| Search Distillation | 能不能把 test-time search 压回参数里 | 把推理时计算变成训练信号 |
| Process Distillation | 能不能蒸馏 reasoning process 而不只是答案 | 让模型学会“怎么想” |
| Weight / Update Distillation | 能不能直接学 `ΔW` | 逼近 optimizer learning / meta-learning |
| Verifier / Critic Distillation | 能不能把“判对错”也独立蒸馏 | 让 solver 和 judge 分工 |
| Capability Distillation | 能不能只蒸馏某一项能力 | 更适合路由、MoE 和专家合并 |
| Dataset Distillation | 能不能连训练样本分布也一起压缩 | 反过来塑造课程和数据选择 |

## 我最看好的几个交叉点

### 1. On-policy Distillation

这个方向最像“把 RL 的缺点补上，但又不完全变成 RL”。

传统蒸馏是 teacher 先生成轨迹，student 再模仿；问题是 student 学到的往往只是 teacher distribution。On-policy distillation 更像是让 student 先自己 rollout，再由 teacher 在这些轨迹上逐步给密集信号。

它的价值在于：student 学的不是老师“平时怎么写”，而是自己真的会走到什么状态、会在哪些位置犯错、需要怎样的修正。

### 2. Self-Distillation + Privileged Information

这类方法更有意思的地方，是“同一个模型，两个信息状态”。

student 处在普通上下文里，teacher 或 critic 处在拥有 ground truth、更多上下文、或者更强判断力的状态里。于是 teacher 不一定直接给答案，而是给 correction、alternative trajectory、token-level preference，甚至直接指出哪一步错了。

这就很像一种 learning with privileged information，而不是传统意义上的大模型教小模型。

### 3. Search Distillation

这个方向我尤其看好。

假设一个 reasoning model 先生成很多条 trajectory，再经过 verifier、reranking、best-of-N、MCTS 或 tree search 找到最好的一条。部署时你当然不想真的跑那么多次搜索，所以最自然的想法就是：

把 inference-time compute 蒸馏回 model parameters。

换句话说，`100x inference compute` 变成 `1x inference compute`，中间靠的不是硬裁剪，而是把 search 的结果学进参数里。

### 4. Process Distillation

这比蒸馏答案更进一步。

答案只告诉你“最后对了什么”，process 则告诉你“中间是怎么对的”。如果把中间的推理步骤、隐藏状态轨迹、uncertainty、verifier state、attention pattern 也纳入蒸馏对象，蒸馏就开始接近 computation transfer，而不只是 knowledge transfer。

### 5. Weight / Update Distillation

`weight learning` 之所以迷人，是因为它让我们开始问另一个问题：

不是只看 `f_{W_t}(x)`，而是直接看 `W_t -> W_t + ΔW_t`。

如果我们能从训练轨迹里学到“什么样的数据会带来什么样的更新”，那蒸馏的对象就不再只是输出，而是更新规则本身。这已经很接近 meta-learning 和 optimizer learning 的交界了。

## 如果要押注 3 到 5 年

我会把优先级排成这样：

| 级别 | 方向 |
| --- | --- |
| S | On-policy distillation |
| S | Search distillation |
| S | Self-distillation + privileged information |
| S | Process / latent distillation |
| S | Verifier / critic distillation |
| S | Weight / update distillation |
| A | Recursive self-improvement |
| A | Capability-specific distillation |
| A | Mutual / reciprocal distillation |
| A | Dataset distillation |
| A | Cross-model / cross-architecture distillation |

这里的意思不是“谁更热闹”，而是“谁更像后训练里真实存在的基础设施”。

## Recursive Self-Improvement 只是外壳

`RSI` 往往被讲得很抽象，但如果把它拆开看，真正的问题只有一个：

每一轮新增的信息从哪里来？

可能来自 search，来自 verifier，来自 environment，来自 tool，来自 ground truth，来自另一个模型，来自 privileged information，来自 synthetic curriculum，甚至来自 weight trajectory 或 self-generated hard cases。

所以我更愿意把 `RSI` 看成蒸馏的一种循环形式：

```text
Model M0
  ↓
generate data
  ↓
train M1
  ↓
M1 generates better data
  ↓
train M2
  ↓
...
```

当这条链路跑顺了，蒸馏就不再是一次性的 teacher/student，而是一个不断自举的信息系统。

## 结尾

蒸馏最近很容易被狭义理解成“模型压缩”，但那只是它最老的一层皮。

如果把蒸馏看成信息迁移机制，那么真正值得研究的，就不只是“老师怎么教学生”，而是：

- 哪类信息值得迁移；
- 应该在什么阶段迁移；
- 是迁移输出、过程、更新，还是判断能力；
- 最后能不能把 search、RL、SFT、verifier 和 data 串成一条闭环。

对我来说，`Search Distillation + On-policy Distillation + Weight/Update Distillation` 是最有意思的交叉点。它们共同指向的不是一个更小的模型，而是一种更会搬运计算的模型。

