# 为什么大模型会陷入重复：从 High Inflow 到结构性循环


大模型开始重复时，表面上像是“同一个词说了太多次”。真实故障往往更严重：生成过程失去向前推进的能力，开始不断确认最近出现的高概率模式，直到外部用 `max_tokens` 强行终止。

开放文本里可能是句段复述，代码里可能是无休止的 `if/elif`、函数壳和断言，翻译里则可能同时出现语言切换和段落循环。

把这些现象放在一起看，repetition 更像一种生成动力学退化：

> 模型滑入了低信息、低变化、但局部概率很高的轨道，而且后续输出不断把它送回同一轨道。

<!--more-->

## 第一层原因：语言分布本身存在 High Inflow

2021 年的理论工作《A Theoretical Analysis of the Repetition Problem in Text Generation》用 Average Repetition Probability 分析重复，并指出一个基础问题：很多不同的当前词，都可能给同一个后继词很高概率。

可以把生成过程想成一张有向图。token 或状态是节点，转移概率是边。如果许多节点都把大概率质量汇入同一小片区域，这片区域就具有 high inflow。模型一旦进入，很容易再次回到刚刚经过的状态，形成循环。

这说明 repetition 并不完全等于“模型训练坏了”。自然语言的局部可预测性、数据中的模板和自回归生成方式，本来就会制造容易回流的区域。论文提出 rebalanced encoding，从表示层面减弱这种高汇入结构。

这个理论模型很简化，但提供了重要起点：重复不是一个孤立 token 的问题，而是状态转移结构的问题。

## 第二层原因：重复会凝结成模型内部特征

《Understanding the Repeat Curse in Large Language Models from a Feature Perspective》进一步把问题从输出分布推进到内部表示。

作者先定位与重复强相关的层，再用 Sparse Autoencoder 提取和操作 repetition features。论文报告，刺激这些特征会放大 token 级和段落级重复，抑制它们则能缓解重复行为。

这类结果意味着，重复并不只是最后一个 softmax 临时做错了选择。训练数据中的模板和循环模式可能已经被编码成稳定特征，在特定上下文中被激活。

但这里必须保留一条边界：相关特征不一定只服务于重复。枚举、格式保持、引用一致性和正常的局部延续也可能依赖相近表示。直接关闭可疑 feature，可能同时伤害模型原本正常的能力。

## 第三层原因：结构化任务会制造“看起来不重复”的重复

普通文本重复比较容易发现，因为字符串经常相同。代码更麻烦：变量名和常量可以不断变化，语法骨架却一直在复制。

《Rethinking Repetition Problems of LLMs in Code Generation》把这种现象定义为 structural repetition。例如模型连续生成很多不同条件的 `elif`，每段文字并不相同，但程序结构已经卡在同一种展开模式中。

这解释了为什么统一的 token repetition penalty 对代码常常不够：

- 相同结构可以使用不同 token；
- 局部代码仍然语法正确；
- 更保守的 temperature、top-k 或 top-p 可能让模型更依赖当前高概率骨架，甚至更难生成 EOS。

论文提出 RPG，也就是 Repetition Penalization based on Grammar。它先用语法规则识别结构性循环，再只降低推动循环继续扩张的关键 token 概率。论文在 CodeRepetEval 上报告优于所比较基线，并在 HumanEval 和 MBPP 上同时改善代码质量。

这里的核心不是 RPG 这一项具体实现，而是诊断单位发生了变化：代码 repetition 应该按语法结构判断，不能只按字符串重合判断。

## 第四层原因：解码会把局部错误变成自强化闭环

生产案例研究《Solving LLM Repetition Problem in Production》记录了业务规则生成、调用关系分析和 PlantUML 生成中的循环输出。

在这些案例中，greedy decoding 每一步只保留当前最高概率路径。一旦这条路径进入循环，后续重复内容又成为下一步上下文，进一步抬高相同模式的概率。模型不是每一步都重新独立犯错，而是在不断给自己的错误轨道追加证据。

论文在这三类案例中比较了多种处理方式，并报告：

- Beam Search 配合 `early_stopping=True` 能覆盖三类案例；
- presence penalty 只对部分案例有效；
- DPO fine-tuning 是更靠近模型偏好的处理方式。

这些结论来自特定生产任务，不能直接变成所有模型的统一配置。它们真正说明的是：终止条件和搜索路径数量会影响模型能否逃离循环，不能只盯着 repetition penalty 一个旋钮。

## 翻译为什么更难修

机器翻译里的重复经常和 language mismatch 同时出现。模型可能一边重复，一边从目标语言退回源语言。

《Mitigating the Language Mismatch and Repetition Issues in LLM-based Machine Translation via Model Editing》尝试定位与这些错误相关的内部组件。作者发现，直接套用已有编辑方法，可能效果有限，也可能损害整体翻译质量；原因是被定位的组件同时参与正常翻译。

他们因而比较不同语言设置下的定位结果，取更稳定的交集后再编辑。这个做法体现了一条很重要的工程原则：越深入模型内部，越需要用正常样本做反事实检查，不能因为某个神经元在坏案例里活跃，就把它当成单一故障开关。

## 一条统一的形成链路

把这些研究拼起来，repetition 可以理解成四步闭环：

1. 语言、数据模板或任务语法提供容易回流的局部结构；
2. 训练把一部分结构编码成稳定的模型偏好和内部特征；
3. 当前任务触发这些高置信模式；
4. 自回归解码把刚生成的模式重新喂给模型，进一步强化下一轮重复。

因此，“把 temperature 调高一点”有时有效，却不是机制完整的解释。它只是在第四步增加逃离概率，没有处理训练偏好、内部表示和任务结构。

## 应该怎样分层治理

### 先判断重复属于哪一种

至少要区分：

- 完全相同的 token 或短语循环；
- 语义相同但表述变化的段落复述；
- 代码、JSON、图描述中的结构性重复；
- 与语言切换、工具调用或终止失败同时出现的复合故障。

检测指标也应随之变化。单纯 n-gram 重复率无法识别不断换变量名的代码骨架。

### 能改解码时，先做低成本止损

可以检查：

- EOS 是否被错误屏蔽或概率长期过低；
- `max_tokens` 是否只是掩盖无法终止；
- greedy、beam 和 sampling 的失败率是否不同；
- presence/frequency penalty 是否只改善表面重复，却伤害任务质量；
- 结构化任务是否需要 grammar-aware 约束。

任何调整都应同时记录重复率、任务正确率、生成长度和终止原因。只让输出变短，不等于修好了重复。

### 能训练模型时，修正偏好而不是只加惩罚

如果坏案例稳定出现，可以构造“正常推进”与“陷入循环”的偏好对，用 SFT 或 DPO 调整模型对轨迹的偏好。训练前还应检查语料中的模板复制、自动生成污染和高度重复结构。

模型级修复通常比单次解码补丁更稳，但也更容易过拟合特定循环模式。评测集需要覆盖正常的枚举、代码复用和格式重复，防止模型学成“任何重复都不允许”。

### 做 feature editing 时保持克制

机制层干预适合研究和难以重训的模型，但必须同时验证：

- feature 在坏案例和正常案例上的激活差异；
- 抑制后重复是否下降；
- 一般生成、翻译或代码能力是否同步下降；
- 效果能否跨 prompt、随机种子和任务保持。

否则很容易把相关性误当成因果开关。

## 结语

LLM repetition 不是一种病，而是一组外观相似的故障。

有些来自概率分布的 high inflow，有些来自模型内部的重复特征，有些由代码语法等任务结构诱发，还有一些是 greedy decoding 和错误终止条件把局部偏好放大成了闭环。

更可靠的处理方式是先定位故障发生在哪一层，再选择最小干预：解码问题改搜索和终止，结构问题加任务感知约束，稳定的模型偏好再进入训练或特征编辑。统一加大 repetition penalty，往往只是让问题换一种形式出现。

## 参考链接

- [A Theoretical Analysis of the Repetition Problem in Text Generation](https://arxiv.org/abs/2012.14660)
- [Understanding the Repeat Curse in Large Language Models from a Feature Perspective](https://arxiv.org/abs/2504.14218)
- [Rethinking Repetition Problems of LLMs in Code Generation](https://arxiv.org/abs/2505.10402)
- [Solving LLM Repetition Problem in Production](https://arxiv.org/abs/2512.04419)
- [Mitigating Language Mismatch and Repetition via Model Editing](https://arxiv.org/abs/2410.07054)

