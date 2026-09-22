# coding agent数据合成-qwen调研


这篇调研聚焦 `coding agent` 训练里最关键的一环：**高质量、可验证、可复现的数据合成**。如果把近两年的公开工作压缩成一句话，更准确的说法是：越来越多系统以真实软件工程活动为种子，再做环境构建、轨迹生成和自动验证，而不是只生成孤立的合成题。

对于 `Qwen` 这类模型，这条路径提供了更接近真实仓库的训练信号。但能否迁移到定位问题、搭环境、写补丁和通过测试，仍要在未见仓库、固定环境和明确测试版本下单独验证，不能只从一个 benchmark 分数推断。

<!--more-->
## 范式演进：从纯合成到“真实-合成”混合数据的转变

近年的公开工作显示，编码数据合成正在从孤立的函数题，转向包含仓库、依赖、测试和执行反馈的“真实—合成”混合流程。这不等于纯合成数据没有价值，而是两类数据承担不同职责：纯合成数据便于控制变量和验证，真实 PR/Issue 则带来噪声、上下文依赖和真实修改习惯。现有研究指出，纯合成任务与真实软件工程任务之间存在分布差异 [[1](https://arxiv.org/html/2510.26130v2)]。SWE-Smith、SWE-Flow 等工作用 TDD 在可控环境中生成任务 [[4](https://arxiv.org/html/2512.17419v1), [52](https://arxiv.org/pdf/2506.09003)]，它们的优势是验证逻辑清楚，边界则是任务可能缺少真实仓库中的噪声和跨文件依赖 [[4](https://arxiv.org/html/2512.17419v1)]。

SWE-bench 的重要变化是直接从 GitHub 上已解决的 Pull Request 中抽取任务，而不是另造一个脱离仓库的题目 [[7](https://www.swebench.com/original.html), [38](https://arxiv.org/html/2507.09108v5)]。但原始 PR 仍需要补齐可复现环境、测试断言和任务边界，才能成为可执行的训练或评测实例。因此，SWE-Synth、SWE-Hub、OpenSWE 和 SWE-Bench++ 等工作都在做同一类转换：把 PR/Issue 变成带环境、轨迹和 verifier 的任务。数据工厂的关键产物不是样本数量，而是每个样本能否被稳定重放和解释。

## 工业界实践：规模化、自动化的“数据工厂”模式

公开的工业项目更容易展示规模化、自动化和端到端基础设施，但“规模大”不等于“数据一定更好”。例如，SWE-Universe 报告从超过 52,000 个 GitHub 仓库生成 807,693 个多语言、可验证实例 [[51](https://arxiv.org/html/2602.02361v1)]；OpenSWE 报告投入约 147 万美元，构建 45,320 个可执行 Docker 环境 [[50](https://arxiv.org/html/2603.13023v1)]。这些数字说明数据合成的基础设施成本很高，也说明比较不同系统时必须同时看任务去重、环境成功率、测试覆盖和下游训练设置。

这类系统通常包含环境自动化、质量保障和训练反馈三个环节。SWE-Bench++ 用模板化 Dockerfile 和迭代反馈构建环境，并报告 Python 仓库产量提升约 137% [[4](https://arxiv.org/html/2512.17419v1)]；OpenSWE 报告在 64 节点集群上自动完成仓库探索、Dockerfile 生成和测试脚本编写 [[50](https://arxiv.org/html/2603.13023v1)]。这些结果应理解为各自管线和基线下的报告值，而不是可直接横向比较的通用倍率。SWE-Bench Atlas 强调环境确定性，SWE-Universe 则报告其数据被用于 Qwen3-Max-Thinking，并在 SWE-Bench Verified 上得到 75.3% [[11](https://openreview.net/forum?id=Gxw1EDSm9S), [51](https://arxiv.org/html/2602.02361v1)]。要判断“数据带来了多少收益”，还需要知道训练数据规模、去重方式、模型版本和评测时间点。

## 学术界探索：基准扩展与合成机制

学术工作通常把问题拆成更容易比较的基准、任务类型和生成机制。Multi-SWE-bench覆盖 Java、TypeScript、Rust 等多语言场景 [[30](https://arxiv.org/html/2504.02605v1)]；Rust-SWE-bench聚焦 Rust 生态 [[10](https://arxiv.org/html/2602.22764v1)]；SWE-Bench Mobile则把评测扩展到移动应用 [[25](https://arxiv.org/html/2602.09540v1)]。这些工作能说明评测覆盖面如何变化，但不能单凭基准名称推出模型能力已经提升。另有诊断性研究指出，SWE-bench 分数可能受到训练数据记忆影响 [[9](https://www.microsoft.com/en-us/research/publication/the-swe-bench-illusion-when-state-of-the-art-llms-remember-instead-of-reason/)]，因此去污染和未见仓库测试仍是必要条件。

合成机制研究关注的是：哪些属性让样本更接近可学习、可验证的软件工程任务。SWE-Synth列出四类约束：缺陷应尽量接近真实开发、任务要能扩展到大型代码库、结果要有自动验证、轨迹要保留定位和修复过程 [[33](https://arxiv.org/html/2504.14757v1)]。这些是数据设计标准，不等于已经验证的统一收益。SWE-World则把部分环境反馈交给 transition model 和 reward model 模拟，以减少对物理 Docker 环境的依赖 [[28](https://arxiv.org/html/2602.03419v1)]。它降低了环境构建的门槛，但仿真器与真实工具链之间的差距仍需要用真实仓库单独检查。

## 技术路径详解：构建可复现软件工程任务的核心工艺

现代数据合成流水线已经形成了一套标准化的、多阶段的工艺流程，其核心目标是将一份看似普通的GitHub PR转化为一个结构完整、可执行且有明确验证标准的软件工程任务。这一过程可以解构为以下几个核心技术环节：

1.  **任务源选择与获取**：这是所有工作的起点。目前最主流的方法是从GitHub的公开Pull Requests (PRs)中挖掘 [[4](https://arxiv.org/html/2512.17419v1), [7](https://www.swebench.com/original.html)]。为了保证数据集的质量和多样性，先进的框架通常会设定一套严格的筛选标准，例如PR的大小、提交历史、仓库的活跃度和社区声誉等，以确保所选任务既有代表性又具备一定的难度 [[11](https://openreview.net/forum?id=Gxw1EDSm9S)]。
2.  **环境合成**：目标是为每个选定的 PR 创建可独立运行、结果尽量确定的测试环境。OpenSWE、SWE-Universe 和 SWE-Bench++主要通过 Dockerfile、依赖解析和迭代测试完成这一步 [[4](https://arxiv.org/html/2512.17419v1), [50](https://arxiv.org/html/2603.13023v1), [51](https://arxiv.org/html/2602.02361v1)]。SWE-World用学习到的环境反馈替代其中一部分物理执行，减少了容器构建需求，但不代表可以跳过真实环境复核 [[28](https://arxiv.org/html/2602.03419v1)]。
3.  **验证逻辑提取**：有了可复现的环境，还要定义什么算“正确”。验证逻辑通常来自 PR 中的单元测试。SWE-Bench++比较父提交（Base）、合并测试但未应用代码变更的状态（Before）和完整 PR 状态（After），再根据测试变化区分 bug 修复与功能请求 [[4](https://arxiv.org/html/2512.17419v1)]。对于不同的测试输出，它先尝试确定性正则匹配，失败后再生成自定义 Python 解析器，并用人工故障检查解析结果。
4.  **数据增强与轨迹合成**：这是提升数据价值的关键。对于那些模型很难解决的难题，可以通过人为提供函数签名、依赖图等“提示”来辅助合成修复轨迹，从而将原本无效的数据转化为有价值的训练样本 [[4](https://arxiv.org/html/2512.17419v1), [11](https://openreview.net/forum?id=Gxw1EDSm9S)]。SWE-Synth框架则更进一步，利用LLM Agent模拟开发者从定位错误、编写代码、调试到最终修复的全过程，生成包含中间步骤的结构化修复轨迹 [[5](https://arxiv.org/abs/2504.14757), [33](https://arxiv.org/html/2504.14757v1)]。这为模型提供了远比单纯的“输入-输出”对更丰富的学习信号。
5.  **数据清洗与过滤**：最后，必须对海量生成的数据进行严格的筛选。OpenSWE的质量中心过滤管道就是一个典型例子，它会表征每个环境的内在难度，并主动过滤掉那些过于简单或环境不可复现的实例，以最大化学习效率 [[50](https://arxiv.org/html/2603.13023v1)]。

| 核心环节 | 关键技术/方法 | 目标与作用 |
| :--- | :--- | :--- |
| **任务源选择** | 爬取GitHub Pull Requests，并基于仓库活跃度、PR大小等标准进行筛选 [[11](https://openreview.net/forum?id=Gxw1EDSm9S)]。 | 确保任务来源的真实性和多样性，为合成高质量数据奠定基础。 |
| **环境合成** | Dockerfile自动化生成、迭代反馈循环、LLM代理学习环境动态 [[4](https://arxiv.org/html/2512.17419v1), [28](https://arxiv.org/html/2602.03419v1)]。 | 创建可复现、确定性的测试环境，使代码修复能够在隔离的沙箱中被验证。 |
| **验证逻辑提取** | 状态差分法、混合日志解析策略 [[4](https://arxiv.org/html/2512.17419v1)]。 | 自动化地识别任务类型（Bug修复 vs. 功能添加），并准确判断代码修复的成败。 |
| **轨迹合成** | 提示引导合成、LLM代理模拟开发者工作流 [[4](https://arxiv.org/html/2512.17419v1), [5](https://arxiv.org/abs/2504.14757)]。 | 生成包含中间步骤的结构化修复轨迹，为模型提供更丰富的学习信号。 |
| **数据清洗** | 难度表征与过滤 [[50](https://arxiv.org/html/2603.13023v1)]。 | 剔除过于简单或无效的任务，保证训练数据的质量和学习效率。 |

## 效果评估与指标演变：量化能力提升与衡量维度深化

效果数字必须连同模型、数据、切分和评测版本一起读。一项 SWE-Bench++ 实验报告 Qwen2.5-Coder-7B 在 SWE-bench Multilingual 上从 5/300 到 11/300；这是同一实验条件下的绝对变化，不能只写成“性能翻倍” [[4](https://arxiv.org/html/2512.17419v1)]。另有报告给出 OpenSWE-32B 在 SWE-bench Verified 上 62.4%、Qwen3-Max-Thinking 为 75.3% [[50](https://arxiv.org/html/2603.13023v1), [51](https://arxiv.org/html/2602.02361v1)]，但它们不是同一模型、同一训练集或同一推理预算，不能构成直接的排行榜比较。

OpenSWE 还报告了 MATH-500 和 SuperGPQA 等非编码任务上的变化，最高约 12 个百分点 [[50](https://arxiv.org/html/2603.13023v1)]。这说明跨任务迁移值得测量，但不能据此推出“真实仓库训练带来通用智能”；还需要知道对照模型、训练混合、样本污染检查和每项任务的推理预算。

随着研究深入，评估也从“能否完成”扩展到“如何完成”。**Pass@k** 是常用的基础指标，但它必须注明尝试次数、预算和采样设置，不能单独代表真实开发质量 [[65](https://arxiv.org/html/2602.10090v1), [136](http://lonepatient.top/2026/02/26/arxiv_papers_2026-02-26)]。FeedbackEval 的 **Repair@k** 观察多轮反馈后的修复率变化，Loc-Bench 则单独测量错误定位能力 [[94](https://arxiv.org/html/2504.06939v2), [138](https://arxiv.org/html/2602.19407v1)]。对训练数据而言，还应记录命令数、编辑量、测试通过率、失败恢复率和 wall-clock 成本。

## 局限与待验证方向：把合成数据做成可比较的实验资产

以 SWE-bench 为代表的数据合成流程仍有两个需要单独验证的风险。第一是**数据污染与评估有效性**：诊断性研究发现，SWE-Bench 分数可能部分来自训练数据记忆，而不是新任务上的推理 [[9](https://www.microsoft.com/en-us/research/publication/the-swe-bench-illusion-when-state-of-the-art-llms-remember-instead-of-reason/)]。如果训练来源与评估仓库重叠，结果就会高估真实能力。SWE-Bench++ 和 SWE-Bench Atlas通过持续加入新 PR 来降低这一风险，但去污染仍需要持续维护，而不是一次性解决。

其次是**成本与可扩展性的矛盾**。OpenSWE报告的百万美元级预算说明，构建大量可执行环境本身就是昂贵环节 [[50](https://arxiv.org/html/2603.13023v1)]。SWE-World、SWE-Universe等工作尝试用自动化或仿真减少这部分成本，但需要同时报告环境成功率、仿真与真实工具链的差异以及单位样本成本。**合成数据的真实性与多样性**也有限：live PR 仍主要反映开源社区实践，不能代替不同规模团队和闭源工程中的协作模式 [[4](https://arxiv.org/html/2512.17419v1)]。

另一个局限是**任务覆盖仍偏向 bug 修复**（Automated Program Repair, APR）[[62](https://arxiv.org/pdf/2505.07372), [127](https://dl.acm.org/doi/10.1145/3631974)]。SWE-Bench++ 已覆盖部分 feature request [[4](https://arxiv.org/html/2512.17419v1)]，但系统设计、架构演进、性能优化和安全增强仍缺少同等成熟的可执行评测。下一步更值得验证的是：自动化环境构建能否保持确定性；仿真环境与真实工具链之间的差距有多大；红队任务是否能覆盖真实失败模式；以及不同任务类型的训练收益能否在去污染的仓库上复现。每个方向都应报告环境成功率、任务去重、轨迹通过率和单位样本成本，而不是只报告生成数量。

## 参考链接

- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770)
- [SWE-bench 官方网站](https://www.swebench.com/)
- [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793)
- [SWE-bench Multimodal: Do AI Systems Generalize to Visual Software Domains?](https://arxiv.org/abs/2410.03859)
- [The SWE-bench Illusion: When State-of-the-Art LLMs Remember Instead of Reason](https://www.microsoft.com/en-us/research/publication/the-swe-bench-illusion-when-state-of-the-art-llms-remember-instead-of-reason/)

