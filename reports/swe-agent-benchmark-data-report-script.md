# Coding Agent 数据与评测版图：汇报稿

## 0. 开场

大家好，今天这部分我想聊 coding agent 的数据和评测版图。

我会围绕三个问题展开。

我们到底应该测什么，才能说明一个模型真的会做软件工程？

现在有哪些现成的大规模 agent 轨迹数据，可以直接用于 SFT、warmup、critic 或 verifier？

哪些 benchmark 或数据集已经提供 Docker 镜像和可复现环境，能被我们接进自己的 pipeline，继续合成 agent rollout？

我想先把结论放在前面：coding agent 的差距，已经不只是一次 patch 写得好不好，而是评测覆盖、可执行轨迹、可复现 SWE 环境，以及持续造数据能力的综合差距。

## 1. 报告框架：三类资产

先把三类资产放到同一张图里看。

Benchmark 是尺子，用来回答“测什么”。比如 SWE-bench Verified、SWE-Bench Pro、SWE-Explore。

Trajectory corpus 是训练材料，用来回答“拿什么训”。比如 Open-SWE-Traces、Orchard、CoderForge、SWE-ZERO-12M。它们提供的是 agent 做任务时的 action、observation、tool call、patch、resolved 标签。

Docker/env 资产更像生产资料。任务能不能真正跑起来、能不能自己采样 rollout，很多时候就卡在这里。比如 SWE-bench 的 `swebench/sweb.eval` 镜像、SWE-rebench V2 的 `swerebenchv2` 镜像、SWE-Pro 的 `jefzda/sweap-images`。

这三类经常共享同一个名字。比如 Open-SWE-Traces 是轨迹集，不等价于镜像集；SWE-rebench V2 是可执行任务池，它既可以被测，也可以成为 rollout seed。

## 2. 评测：从最终修复到子能力诊断

先看评测。

老牌核心 benchmark 是 SWE-bench 系列，SWE-bench 最早在 arXiv 上是 2023 年 10 月 10 日。它把模型从函数级编程题，推到了真实 GitHub issue resolving：给仓库和 issue，让 agent 修改代码，然后用测试判断是否解决。

这些老牌 benchmark 在大厂模型发布里仍然很重要。SWE-bench Verified 适合 sanity 和横向对比；SWE-Bench Pro / SWEAP 是 2025 年 9 月 21 日的工作，更难，更接近工业级长周期任务。

页面左侧放的是这些常测基准，大概占三分之一。它们不一定最新，但每次看模型发布、产品能力和回归测试都绕不开。

右侧放近年新增和子能力评测。

SWE-bench Verified 一度是最重要的标准尺子，因为它由人工复核，更稳定。但现在 frontier coding agent 已经接近刷穿 Verified，而且存在污染和测量失真的问题。所以工业界开始转向更难的任务，比如 SWE-Bench Pro、SWE-Lancer。

SWE-Bench Pro，也就是 SWEAP，更接近长周期和工业级软件工程任务。SWE-Lancer 则来自真实 freelance 软件工程任务，信号更接近“开发者愿不愿意为这个结果付钱”。

除了最终 repair，学界最近开始把能力拆得更细。

SWE-Explore 是 2026 年 6 月 5 日的工作，测的是仓库探索能力：给定 repo 和 issue，让 agent 在有限 line budget 下找出 relevant code regions。它不直接测最终 patch，而是测 agent 有没有先找到对的上下文。

MULocBench / Multi-CoLoR 这类工作测 localization，重点是文件、函数、行级定位。

SWE-QA / SWE-QA-Pro 测 repo-level QA，关注多文件、多跳依赖和架构理解。

SWE-Skills-Bench 是 2026 年 3 月 16 日的工作，测 skill 文档是否真的提升 agent，而不是把 skill 当 prompt 装饰。

SWE-Doctor 是 2026 年 7 月 1 日刚挂到 arXiv 的工作，它进一步把测试失败转成 runtime diagnosis，用 bug reproduction tests 的结果指导修复。

所以这条路线大概是：

```text
函数级代码生成
-> 真实 issue repair
-> 长周期工业任务
-> 真实 SWE workflow
-> 探索、定位、诊断、技能、长期维护等子能力
```

这个变化对我们很重要。rollout 数据不应该只存最终 patch，中间的文件探索、命令执行、测试失败、定位过程，本身都可以变成训练信号。

## 3. 现成轨迹数据集

接着看现成轨迹数据。

这类数据不是 benchmark 本身，而是 agent 做任务时留下的 traces。它们可以用于 SFT、agentic warmup、critic、verifier，也可以给我们的 schema 做参考。

Open-SWE-Traces 是这里很重要的一份，论文是 2026 年 6 月 14 日提交到 arXiv。它有 207,489 条轨迹，来源是 SWE-rebench-V2，覆盖多语言，teacher 是 MiniMax-M2.5 和 Qwen3.5，scaffold 包括 OpenHands 和 SWE-agent。它的价值在于多语言、resolved 三值标签，以及 thinking / non-thinking teacher 组合。

Microsoft Orchard 也很值得关注，arXiv 时间是 2026 年 5 月 14 日。它有 107,185 条轨迹，同样使用 MiniMax-M2.5 和 Qwen3.5，但 harness 是 mini-swe-agent 和 OpenHands，来源包括 SWE-rebench 和 Scale-SWE。它的特点是保留 unresolved 轨迹，这对 credit assignment 和 failure analysis 很有价值。

SWE-ZERO-12M 是另一个方向，相关的 SWE-ZERO 到 SWE-HERO 论文是 2026 年 4 月 2 日提交到 arXiv。它非常大，有 12.29M rollouts，约 112B tokens，覆盖 122k PR。但它是 execution-free：不建 Docker、不跑测试、不 consult verifier。它适合作 agentic mid-training，让模型先学会 shell、grep、读文件、仓库导航和 action/observation 格式；但它不适合作为 resolved 成功轨迹。

CoderForge-Preview 提供 258k test-verified trajectories，51k tasks，并且 pass/fail 都有。它能说明 execution-backed SFT 数据可以显著提升 SWE-bench 表现。

Nebius 的 SWE-rebench OpenHands trajectories 和 SWE-agent-trajectories，则和我们自己的 pipeline 更容易对齐，因为它们本身就来自 SWE-rebench / SWE-agent / OpenHands 这条线。

所以公开轨迹数据很适合做 agentic prior 和格式 warmup。但这些数据通常绑定特定 teacher、scaffold 和 action schema。真正要和自己的模型、产品、训练栈对齐，还是要靠自己的 pipeline 继续生成 execution-backed rollout。

## 4. 可运行环境：哪些能接进我们的 rollout pipeline

再往下看可运行环境。

这一层关系到我们能不能自己合成 agent rollout。只有 SWE 任务池有镜像、Dockerfile 或稳定可复现环境，我们才能把它接入 K8s，跑 OpenHands、Claude Code 或 SWE-agent，采集轨迹。

当前最优先的是三类。

SWE-bench Verified 规模不大，但非常稳定，适合 sanity、gold smoke、scorer 校验。

SWE-rebench V2 有约 32K executable tasks，配套 `swerebenchv2` 镜像，是目前最适合大规模 rollout 的主力池。它还有 120K+ 额外 PR tasks，但那部分没有现成镜像，只有 install instructions，接入成本更高。

Scale-SWE 也应该单独看。它是 2026 年 2 月 10 日的工作，完整口径是 100K verified instances，开源部分常见统计约 20.2K。它和 Orchard、SWE-rebench 这条线关系很近，更适合放在“大规模可构建/可验证任务池”这一类，接入前要先抽样确认 image_url、依赖和测试是否稳定。

SWE-Bench Pro / SWEAP 有 `jefzda/sweap-images`，难度更高，更适合作为 hard set 和工业 benchmark 对齐。

此外，还有几类之前容易漏掉的环境资产。SWE-bench-Live 有 1,319 个 2024 年之后的 live issue tasks，每个任务配 dedicated Docker image，适合做 fresh holdout。Multi-SWE-bench、SWE-PolyBench、SWE-bench-java 主要补多语言和 Java 工程栈。SWE-Dev 把任务从 bug fix 推到 feature development，并提供 runnable environment 和 executable unit tests。SWE-Perf 则测性能优化任务，带 performance tests 和 executable environments。

SWE-Gym、R2E-Gym、SWE-smith 也都有可运行环境或镜像资产，可以作为补充任务池。

OpenSWE / daVinci-Env 提供大量 Dockerfiles、evaluation scripts 和 infra，规模很大，但接入时需要确认镜像 registry 和可复现性。

## 5. 我们的 pipeline：SWE rollout factory

基于上面几部分，我们的 pipeline 可以定位成 SWE rollout factory。

它不是重新发明 SWE-bench，也不是单纯跑 benchmark 分数，而是把已有可执行任务池变成持续产生训练轨迹的系统。

流程可以概括为：

```text
Benchmark seeds
-> image resolver
-> K8s Pod
-> agent scaffold
-> trace capture
-> verifier
-> training views
```

Benchmark seeds 可以来自 SWE-bench Verified、SWE-rebench V2、Scale-SWE、SWE-Pro、R2E-Gym。

Image resolver 负责镜像重写、镜像存在性检查、网络权限分类。

K8s Pod 由 ltp-docker 负责启动，上传脚本，准备 repo 环境。

Agent scaffold 可以是 OpenHands、Claude Code、SWE-agent，但要尽量统一 action schema，避免训练时把 scaffold 差异误当成模型能力。

Trace capture 要记录 spans、token/logprob、stdout/stderr、diff、测试命令、工具调用。

Verifier 不只看最终 pass/fail，还要包含 gold patch filter、resolved scorer、flaky 重跑、弱测试检测、patch cheating 检测。

最后导出多种 training views：SFT triplet、preference pair、critic data、verifier data、RL replay。

## 6. 后续改进方向

短期可以先做四件事。

把 SWE-Pro/SWEAP、Scale-SWE 和多语言 SWE 任务池加入统一 run manifest。

给每个 seed 记录 image、语言、repo、issue、验证状态和失败原因。

成功和失败轨迹都保留标签。失败轨迹不是垃圾，它可以用于 critic、failure classifier 和 credit assignment。

按 scaffold 拆分统计，避免 OpenHands、Claude Code、SWE-agent 的格式差异污染训练。

中期可以继续往三个方向走。

从成功轨迹里蒸馏 SWE-Explore 式 code-region labels，训练 context selector。

引入 SWE-Doctor 式 runtime diagnosis，把测试失败转成中间监督。

建立 trajectory quality audit，对定位失败、命令失败、环境失败、patch 作弊、过早结束做细粒度标签。

## 7. 合成数据还能怎么合成

在讲 Docker 资产之前，我想先补一个问题：除了 Docker rollout，SWE 数据还能怎么合成？

Docker rollout 是最硬的 execution-backed 路线，因为它能真正跑测试、验证 patch。但它成本也最高。很多时候，我们还需要成本更低、粒度更细的合成信号。

一种做法是 PR 反演。从真实 merged PR 中隐藏 patch，保留 issue、commit message、diff 上下文和测试变化，反向构造修复任务。

也可以走测试驱动。先合成或抽取 failing tests，再让 agent 生成实现或修复。功能添加、边界条件、回归测试和 verifier 数据都能从这里来。

还有 bug 注入。对真实仓库做接近真实错误的扰动，比如条件反转、API 误用、边界错误、类型错误，再用原测试或新增测试验证。

静态结构也能造监督。用 diff、调用图、测试覆盖、stack trace、import graph 生成文件、函数、行级相关区域标签，训练 explorer 和 retriever。

teacher 轨迹蒸馏是另一条路线。让强 teacher 在无 Docker 或弱验证环境里生成 action/observation 轨迹，清掉格式、重复和明显错误后，用来做 agentic warmup。

失败轨迹也不一定要丢。我们可以标注它到底是定位失败、命令失败、测试误读、环境失败，还是 patch 过拟合，然后拿去训练 critic、preference 和 credit assignment。

日志也能变成诊断数据。从 pytest、build、CI、runtime logs 中抽出“错误现象 -> 根因 -> 下一步动作”，形成类似 SWE-Doctor 的诊断层。

CI 反馈任务则可以从 CI script、失败测试、build logs 和依赖变更里生成，用来训练 agent 读失败原因、改代码并重新验证。

这些方法的分工很清楚：execution-free 数据打底行为先验；静态定位和日志诊断补中间监督；Docker rollout 负责最终 resolved 信号和 harness 对齐。

## 8. 结尾

最后收一下。

SWE agent 数据和评测正在分成三条线：更真实的 benchmark，更大的 trajectory corpus，更可复现的 Docker/env 资产。

公开大数据可以帮助我们快速获得 agentic prior；现成镜像池可以帮助我们规模化生产 execution-backed rollout；而我们的 pipeline 的价值，是把这两者接起来，持续生成和自己 harness、模型、训练目标对齐的数据。

所以这件事的目标不是再收集一个静态数据集，而是建立一个可以持续迭代的 SWE rollout factory。
