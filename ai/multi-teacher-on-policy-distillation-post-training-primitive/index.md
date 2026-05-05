# 读 MOPD：后训练正在从单一 RL 走向多教师能力合并


这篇文章最值得注意的地方，不是它又给后训练加了一个新缩写，而是它指出了一个正在变清晰的趋势：

**后训练的主线，可能正在从“用一个奖励函数把模型直接 RL 到更强”，转向“先训练多个能力专家，再用 on-policy distillation 把这些能力合并回一个主模型”。**

如果说 SFT 是教模型模仿答案，DPO/RL 是教模型偏向更好的行为，那么 `Multi-Teacher On-Policy Distillation` 更像是在回答另一个问题：

**当我们已经用不同数据、不同奖励、不同阶段训练出了多个强 teacher，怎样把它们稳定地压回一个 student，而不是靠简单模型融合或离线蒸馏丢掉能力？**

<!--more-->

## 为什么这件事重要

过去两年，LLM 后训练的叙事很容易被简化成一条线：

```text
SFT -> Preference Optimization -> RL
```

这条线当然成立。尤其在 reasoning model 里，`GRPO`、`RLVR`、verifier-based RL 这些方法，确实让模型在数学、代码、长链推理上获得了明显提升。

但这条线也有一个问题：它默认最后的提升主要来自一个统一的 RL 过程。真实工程里，事情通常没有这么干净。

一个强模型往往不是被单一目标函数推出来的，而是由很多阶段共同拼出来的：

- 数学 teacher 擅长长推理。
- 代码 teacher 擅长调试和工具使用。
- 通用对话 teacher 擅长表达和遵循指令。
- 某些阶段的 checkpoint 更稳，某些阶段的 checkpoint 更激进。
- RL 之后的模型能力更强，但也可能出现格式漂移、语言退化或过拟合奖励。

这时真正的问题就变成了：

**RL 可以制造专家，但不一定适合直接制造最终产品模型。**

`MOPD` 的价值就在这里。它把后训练拆成两层：第一层用 RL、SFT 或领域数据训练出多个 teacher；第二层让 student 在自己的分布上生成轨迹，再从这些 teacher 中学习。也就是说，RL 不再只是终点，而变成了制造 teacher 的中间工艺。

## 从离线蒸馏到 on-policy 蒸馏

普通蒸馏最自然的做法，是拿 teacher 生成一批答案，然后让 student 去模仿：

```text
teacher 生成数据 -> student 在这些数据上训练
```

这看起来简单，但对后训练来说有一个很大的问题：这些数据来自 teacher 的分布，不来自 student 自己的分布。

模型真正部署时，是 student 自己一步步生成。只要前面某个 token 偏了一点，后面的状态就可能进入 teacher 数据里从来没出现过的区域。这个问题本质上就是 off-policy mismatch。

`On-Policy Distillation` 的关键变化是：不让 student 只模仿 teacher 生成过的答案，而是先让 student 自己生成轨迹：

```text
student 自己采样回答
teacher 在 student 的轨迹上给出 token-level 分布
student 学习 teacher 在这些状态下会怎么走
```

形式上，可以把它理解成：

$$
y \sim \pi_\theta(\cdot \mid x)
$$

然后在 student 访问到的前缀状态上，让 teacher 提供下一步分布：

$$
\pi_T(\cdot \mid x, y_{<t})
$$

student 优化的是这些 on-policy 状态上的分布差异，而不是只背 teacher 的完整答案。

这件事的直觉非常重要：**teacher 不是只告诉 student 标准答案是什么，而是在 student 自己会走到的路口，告诉它更好的下一步怎么选。**

## Multi-Teacher 的意义：不是一个老师，而是一组专家

如果只有一个 teacher，OPD 已经能缓解 off-policy mismatch。但真实后训练里，往往没有一个 teacher 在所有维度上都最好。

这就是 `Multi-Teacher` 的必要性。

一个 student 可能需要同时吸收：

- reasoning teacher 的推理深度；
- code teacher 的执行稳定性；
- chat teacher 的表达与安全边界；
- math teacher 的答案校验习惯；
- early checkpoint 的语言自然度；
- late checkpoint 的任务成功率。

MOPD 的核心不是“多找几个模型投票”，而是把多个 teacher 当成多个能力源。student 在自己的生成分布上探索，系统再决定当前样本、当前领域，甚至当前 token 更应该向哪个 teacher 学。

这比简单 ensemble 更适合作为后训练 primitive。ensemble 是推理时把多个模型绑在一起，成本高，也不一定能部署；MOPD 是训练时把多个 teacher 的行为压进一个 student，最终仍然得到一个单模型。

## 它和 RL 到底是什么关系

我觉得最容易误解的一点，是把 MOPD 看成 RL 的替代品。更准确地说，它像是 RL 的下游接口。

RL 的优势是能用 outcome signal 优化模型行为：

```text
答案对不对；
测试过没过；
verifier 给多少分；
工具调用是否成功；
推理轨迹是否满足规则。
```

但 RL 也有几个老问题：

- 奖励信号稀疏，训练不稳定；
- 对 reward model 或 verifier 的偏差很敏感；
- 容易牺牲通用语言能力；
- 不同任务的 reward 目标很难混在一个训练阶段里；
- 多个 RL checkpoint 之间能力互补，但很难直接合并。

MOPD 的位置刚好在这里：

```text
RL 负责把某些 teacher 推到更强；
MOPD 负责把多个 teacher 的能力合并进 student。
```

所以它不是反 RL，而是把 RL 从“最终成品训练器”改造成“专家能力生产器”。这其实更接近现代模型工程的真实形态：先分工训练，再统一蒸馏。

## 为什么说它是新的后训练原语

一个方法能不能称为 primitive，关键不在于它复杂不复杂，而在于它能不能成为很多 pipeline 的基础积木。

MOPD 之所以值得被看成后训练 primitive，是因为它解决的是一个反复出现的结构性问题：

**多个训练阶段、多个专家模型、多个奖励目标之间的能力合并。**

在更早的后训练流程里，这件事常常被粗暴处理：

- 直接混合数据做 SFT；
- 直接选择最强 checkpoint；
- 用一个统一 reward 做 RL；
- 用离线 teacher 数据蒸馏；
- 推理时做 ensemble 或 routing。

这些方法都能用，但都有明显代价。MOPD 更像是把能力合并这件事系统化了：

```text
teacher 可以来自不同阶段；
teacher 可以来自不同领域；
teacher 可以来自不同 RL 目标；
student 训练时保持 on-policy；
最终产物仍然是一个单模型。
```

这正是它和后训练范式相关的地方。未来的 post-training pipeline 可能不再是单线条，而更像一个能力工厂：

```text
base model
-> domain SFT
-> RL specialists
-> multi-teacher on-policy distillation
-> final instruction / safety polish
-> deployable model
```

## 几篇技术报告共同说明了什么

这篇内容把 `MiMo`、`GLM-5`、`Nemotron-Cascade 2`、`DeepSeek-V4` 等报告放在一起看，意义不在于每篇都用了完全相同的实现，而在于它们共同指向同一个趋势：

**后训练正在从单模型单阶段优化，转向多阶段、多 teacher、多能力源的系统工程。**

`MiMo` 这类 reasoning report 说明，小模型也可以通过强后训练和蒸馏获得很强的推理能力。这里重要的不是“蒸馏小模型”这个老概念，而是 teacher 的能力来自更复杂的后训练过程。

`GLM-5` 里强调的 cross-stage/on-policy distillation，则更像是在处理阶段之间的能力迁移问题：某些阶段带来推理能力，某些阶段保留通用能力，最后要把它们重新整合。

`Nemotron-Cascade 2` 这类工作把 cascade、teacher routing、distillation 放到一套系统里看，说明 teacher 不一定只是一个固定模型，也可以是一组按能力分工的模型系统。

`DeepSeek-V4` 报告中关于 specialist training 和 distillation 的设计，也说明了同一件事：强模型的后训练越来越像“专家生产 + 能力合并”，而不是单次 RL 打到底。

这些报告放在一起，MOPD 就不只是某个孤立技巧，而像是对一个共同工程模式的命名。

## 我觉得最核心的 insight

如果把这篇内容压缩成一句话，我会写成：

**RL 负责发现和放大能力，MOPD 负责把能力稳定地装回主模型。**

这句话背后有三个判断。

第一，未来不会只有一个 teacher。因为能力是多维的，数学、代码、对话、安全、工具使用、长上下文，各自最优的模型状态可能并不相同。

第二，未来不会只靠离线蒸馏。因为 student 最需要指导的地方，不是 teacher 原本会怎么写，而是 student 自己会在哪些状态里犯错。

第三，未来的后训练会越来越像系统工程。RL、distillation、routing、verifier、specialist、data filtering 都会变成同一条流水线里的不同模块。

## 潜在问题

MOPD 也不是没有代价。

首先，多 teacher 带来新的选择问题：当前样本该听哪个 teacher？如果 teacher 之间冲突，应该平均、加权，还是让 router 决定？如果强 teacher 在某些领域其实有偏差，student 可能会把偏差也继承下来。

其次，on-policy distillation 的计算成本更高。student 要先 rollout，teacher 还要在 student 的轨迹上做 forward。如果 teacher 很大、数量很多，这个过程会非常贵。

第三，MOPD 对 teacher 的质量依赖很强。它能合并能力，也可能合并坏习惯。如果没有 verifier、数据过滤和 teacher selection，multi-teacher 可能只是把噪声混得更复杂。

最后，它还没有完全回答一个更深的问题：

**当 teacher 的分布和最终任务奖励冲突时，student 应该相信 teacher，还是相信 reward？**

这也是 MOPD 和 RL 未来最值得继续结合的地方。一个合理方向是：teacher 给局部行为先验，reward/verifier 给最终结果约束。前者让训练稳定，后者防止模型只学会模仿而不真正优化任务。

## 结语

我觉得 MOPD 的价值，不在于它替代了 RL，而在于它重新安排了 RL 在后训练流水线里的位置。

以前我们容易把 RL 想成最后一锤：

```text
把模型 RL 到最强，然后上线。
```

但 MOPD 暗示的未来更像：

```text
用 RL 训练多个专家；
用 on-policy distillation 合并专家；
再用 verifier 和少量 RL 做最终校准。
```

这会让后训练从单一目标优化，变成一套更模块化的能力生产系统。

如果这个趋势继续发展，未来我们讨论 post-training 时，可能不会只问“用了什么 RL 算法”，而会问：

```text
有哪些 teacher？
teacher 是怎么训练出来的？
student 的 on-policy 数据怎么采？
teacher 冲突怎么处理？
能力合并后怎么验证？
哪些失败样本会回流到下一轮 teacher 训练？
```

这才是我觉得 `Multi-Teacher On-Policy Distillation` 最有意思的地方：它把后训练的重心，从单个模型的奖励优化，推向了多模型、多阶段、可复用的能力合并。

## 参考链接

- [Multi-Teacher On-Policy Distillation: A New Post-Training Primitive](https://yumoxu.notion.site/multi-teacher-on-policy-distillation)
- [On-Policy Distillation, Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [MiMo: Unlocking the Reasoning Potential of Language Model](https://arxiv.org/abs/2601.02780)
- [GLM-5 Technical Report](https://arxiv.org/abs/2602.15763)
- [Nemotron-Cascade 2 Technical Report](https://arxiv.org/abs/2603.19220)
- [DeepSeek-V4 Technical Report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)

