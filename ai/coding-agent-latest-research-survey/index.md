# Coding Agent 最新论文与工程调研


这轮调研的核心问题是：**为什么 GPT/Codex、Claude Code 这类 coding agent 突然变得这么能干？**

我的结论先放前面：它们不是单靠“模型会写代码”变强的，而是被一整套新的训练与工程范式推上去的：

1. **数据从代码片段变成真实软件工程任务**：GitHub issue、PR、测试、依赖、Docker 环境、终端日志、代码审查意见、失败轨迹都成了训练资产。
2. **训练从 next-token code completion 走向可验证交互**：SFT 学轨迹，RL 学“跑测试、看失败、改补丁、再验证”的闭环，critic/verifier 学会挑更可能通过的 patch。
3. **terminal 成了核心训练环境**：真实 agent 不是只输出 diff，而是在 shell 里安装依赖、读日志、运行测试、grep/rg、编辑文件、回滚、调试。
4. **长上下文不只是 1M token，而是 context engineering**：强 agent 会把仓库当文件系统操作，用搜索、摘要、记忆、压缩、多窗口初始化来管理上下文。
5. **harness/runtime 的设计极其关键**：同一个底座模型，换工具接口、权限模型、sandbox、上下文拼装、测试反馈方式，SWE-bench 分数和真实可用性都能差很多。

还有一个必须诚实说清楚的点：**OpenAI 和 Anthropic 没有公开完整训练配方**。但它们的官方 blog/system card 已经泄露了足够多的方向信号。OpenAI 明确说 GPT-5-Codex 类模型是针对 agentic coding 优化，并使用真实 coding tasks 的强化学习，让模型学会贴近 human PR preference、遵循指令、迭代跑测试直到通过。Anthropic 则把 Claude Code 的进步拆成 context engineering、工具设计、sandbox、auto mode、long-running harness、多 agent 并行和 eval-aware 开发。把这些和开源论文拼起来，大致能还原“前沿 coding agent 为什么强”的工程轮廓。

<!--more-->

## 一张总图：coding agent 能力栈

可以把现在的 coding agent 能力拆成六层：

| 层级 | 代表问题 | 关键资料 |
| :--- | :--- | :--- |
| 基础代码模型 | 会不会补全、理解 API、跨文件推断 | StarCoder2、DeepSeek-Coder-V2、Qwen2.5-Coder、ExecRepoBench |
| 仓库级任务数据 | 有没有真实 issue/PR/test/env | SWE-bench、SWE-Gym、SWE-smith、R2E-Gym、SWE-Fixer |
| 交互轨迹训练 | 会不会像开发者一样迭代 | SWE-Gym trajectories、SWE-Dev、DeepSWE、SWE-RL、RepoForge |
| terminal/CLI 能力 | 会不会用 shell 和执行环境 | Terminal-Bench、CLI-Gym、Terminal-World、OSWorld、CUA-Gym |
| 长上下文/上下文工程 | 会不会在大仓库里找对上下文 | RepoBench、RepoQA、Long Code Arena、LongCodeU、Anthropic context engineering |
| harness/runtime | 模型如何被安全、高效地放进真实工作流 | SWE-agent ACI、OpenHands、Codex agent loop、Claude Code sandbox/auto mode |

这六层里，最容易被低估的是第 3、4、6 层：**轨迹、terminal 和 harness**。因为它们不像模型参数那么显眼，但真正决定 agent 能不能端到端完成任务。

## SWE 数据合成：从“题库”到“可执行软件工程世界”

SWE-bench 的范式是整个领域的分水岭。它把评测从 HumanEval/MBPP 那种函数题，推到真实 GitHub issue：给模型一个仓库和一个 issue，让它产出 patch，再用测试判断是否解决。原始 SWE-bench 收集了 2,294 个 Python 仓库任务；SWE-bench Verified 由 OpenAI 人工复核出 500 个更可靠子集；后来又出现 SWE-Bench Pro、Multi-SWE-bench、Rust-SWE-bench、SWE-bench Multimodal、DeepSWE 等更难、更长、更抗污染的数据。

但 SWE-bench 同时暴露了两个问题：

- **训练数据太少**：真实、可执行、可验证的软件工程任务太难构造。
- **评测很快被刷穿和污染**：OpenAI 在 2026 年已经公开建议不要再把 SWE-bench Verified 当 frontier coding 的主要信号，而转向更难、更抗污染的 SWE-Bench Pro。

所以 2025-2026 年最重要的方向就是：**如何规模化生产可验证 SWE 环境和轨迹**。

典型流水线大概长这样：

1. 从 GitHub issue/PR、开源仓库、freelance 任务、真实 terminal workflow 或合成项目里采种子。
2. 构造可复现环境：Dockerfile、依赖锁定、build/test 命令、初始文件状态。
3. 生成任务说明：issue 摘要、失败日志、需求文档、错误信息、测试目标。
4. 生成或抽取 verifier：单元测试、集成测试、program-based evaluator、reward function。
5. 跑 agent 采样轨迹：成功轨迹、失败轨迹、多轮修复、测试反馈、工具调用序列。
6. 训练模型或 verifier：SFT 学行为格式，RL 学通过测试，critic/reranker 学选择 patch。
7. 过滤和去重：环境是否稳定、测试是否过弱、问题是否泄漏答案、patch 是否作弊。

代表项目里，SWE-smith 用任意 Python 仓库合成大量会破坏测试的任务，生成 50k+ instances；SWE-Gym 提供 2,438 个真实 Python 任务和 agent trajectories；R2E-Gym 用 SWE-Gen 构造 8K+ 可执行环境；CLI-Gym 把 Dockerfile/环境历史反向变成 terminal 任务；Terminal-World 用 agent skills 合成 terminal training environments；CUA-Gym 则把类似思路推广到 computer-use agents。

这背后的共同思想是：**训练 coding agent 的关键不是“更多代码”，而是更多可验证的状态转移**。也就是：在某个仓库状态下，agent 做了什么命令，观察到什么失败，如何定位，如何修改，如何验证，最后奖励是什么。

## 训练方法：SFT、RL、verifier 和 test-time scaling 的组合拳

现在比较可信的训练路径不是单一路线，而是组合：

### 1. 代码预训练仍然重要，但不够

Qwen2.5-Coder、DeepSeek-Coder-V2、StarCoder2 这类 code LLM 把基础代码能力做扎实：大量源码、文档、PR、notebook、issue、数学/推理文本、多语言数据、FIM/补全/修复/解释任务。它们解决的是“会不会读写代码”的底座问题。

但到了真实 SWE 任务，瓶颈常常不是语法，而是：

- 找不到该改哪几个文件；
- 不会从测试失败反推 bug；
- 跑不通依赖环境；
- patch 过拟合弱测试；
- 做完不验证；
- 长任务中忘记早期约束。

所以后训练必须进入 agent 轨迹。

### 2. SFT 学“开发者动作分布”

SWE-Fixer 的 retrieve-then-edit、SWE-Dev 的 agent trajectory 数据、Kimi-Dev 的 Agentless skill prior、SWE-Gym 的开放轨迹，都在证明一件事：模型需要先学会一套基础软件工程动作：

```text
理解 issue -> 检索相关文件 -> 定位函数/类 -> 生成 patch -> 运行测试 -> 根据错误调整 -> 输出最小可审查 diff
```

SFT 的价值不是让模型记答案，而是让它学会**行为先验**：什么时候 grep，什么时候打开文件，什么时候运行测试，什么时候不要大范围重构。

### 3. RL 学“在环境里拿奖励”

DeepSWE、SWE-RL、R2E-Gym、RepoForge、Terminal-World、CLI-Gym 都指向 RLVR/RL with verifiable reward：只要任务能被测试或 reward function 验证，就可以让 agent 在真实/模拟环境里反复尝试，从 pass/fail 学策略。

这里和数学 RL 有一个很大区别：SWE RL 是长上下文、多轮、工具调用、环境反馈，而且奖励稀疏。一个任务里可能有几十到几百步，失败原因可能是定位错、命令错、patch 错、没跑测试、依赖没装好。于是很多论文开始引入：

- 分阶段 credit assignment；
- verifier/critic 选择候选 patch；
- hybrid verification：执行测试 + 非执行式静态/LLM verifier；
- rejection sampling / best-of-N / parallel attempts；
- 训练成功轨迹，也保留失败轨迹做反例或偏好数据。

### 4. Test-time scaling 已经成为显性能力

OpenHands 的 critic model、SWE-Gym 的 verifier、Nebius 的 search、DeepSWE 的 TTS、Codex/Claude 的多 session 工作流，本质都是：**给 agent 更多尝试、更多验证、更多挑选机制**。

一个很现实的判断是：前沿 coding agent 的强，不只是单次输出强，而是它们有能力在受控预算内：

- 并行尝试多个修复方向；
- 保留和比较多个 patch；
- 用测试、lint、review model 过滤；
- 在失败时自动缩小问题；
- 在长任务里持续维护状态。

这也是为什么“模型分数”越来越难和“产品 agent 体验”分开看。

## Terminal 数据训练：coding agent 真正的手脚

用户特别提到 terminal 数据训练，这确实是 2026 年很关键的一条线。

Terminal-Bench 2.0 把 agent 放进真实命令行环境，任务包括编译代码、训练模型、配置服务、调试系统问题等，每个任务有独立环境、人类解法和测试。CLI-Gym 更进一步：通过“环境反演”生成 1,655 个 environment-intensive 任务，并用成功轨迹微调 LiberCoder，在 Terminal-Bench 上大幅提升。Terminal-World 则把 skill、precondition、environment state、teacher trajectory 合成在一起，生产 5,723 个训练环境。

这说明 terminal 能力不是“让模型背一些 bash 命令”这么简单，而是要训练：

- 命令选择：什么时候用 `rg`、`pytest`、`npm test`、`git diff`、`python -m`；
- 输出解析：从长日志中抓关键错误；
- 环境修复：依赖、路径、权限、版本、端口、配置；
- 迭代策略：一次失败后换更小测试、加日志、检查 diff；
- 安全边界：哪些命令需要权限，哪些命令不能自动执行；
- 状态管理：当前目录、环境变量、已改文件、测试覆盖范围。

OpenAI Codex 和 Claude Code 的产品形态都把 terminal 放在中心位置：Codex CLI/云沙箱，Claude Code terminal/IDE/sandbox/auto mode。我的推断是，前沿模型的后训练数据里一定有大量“shell command -> observation -> next action”的真实或合成轨迹，否则很难解释它们在长时间工程任务里的稳定性。

## 长上下文：不是把整个仓库塞进去，而是让 agent 会经营上下文

长上下文对 coding agent 很重要，但“1M token 能装下整个仓库”不是终点。真实问题是：模型能不能在长任务里持续知道什么重要、什么已经验证、什么应该遗忘。

长上下文相关资料可以分三类：

1. **仓库级理解 benchmark**：RepoBench、RepoQA、Long Code Arena、LongCodeU、ExecRepoBench。
2. **agent 外部化上下文处理**：Coding Agents are Effective Long-Context Processors 提出一个有意思观点：与其让模型在注意力里被动处理长文本，不如让 coding agent 用文件系统、脚本、搜索、摘要来主动处理长上下文。
3. **工业 context engineering**：Anthropic 的 context engineering、effective harnesses、advanced tool use、MCP code execution；OpenAI 的 Codex agent loop、context window management、Codex harness。

强 coding agent 的长上下文能力，更像这样：

```text
先扫仓库结构 -> 建立 task map -> 只打开相关文件 -> 运行验证 -> 把中间发现写入 scratch/context -> 必要时压缩历史 -> 继续下一轮
```

这也是 Claude Code 和 Codex 都强调“context compaction / memory / initializer / repository knowledge / tool discovery”的原因。它们不是简单扩大窗口，而是在做一套 agent 操作系统里的上下文管理。

## GPT/Codex 和 Claude Code 为什么强：可公开推断的训练/工程配方

以下是基于官方资料和开源研究的推断，不是泄露配方。

### OpenAI/Codex 线索

OpenAI 已经公开了几个关键信号：

- Codex 是云端/本地 coding agent，每个任务跑在独立 sandbox/container，预加载仓库和开发环境。
- o3/o4-mini Codex addendum 描述了无互联网的 cloud container trajectory：环境设置后关闭网络，模型开始执行任务。
- GPT-5-Codex system card addendum 明确说它是针对 Codex 中 agentic coding 优化，使用真实 coding tasks 的 RL，目标包括 human style、PR preference、严格遵循指令、迭代运行测试直到通过。
- Codex agent loop 文章把 harness 拆成：模型推理、tool schema、上下文构造、工具调用、观察结果、继续循环。
- OpenAI 2026 年关于 SWE-bench Verified 的文章承认旧 benchmark 污染和测量失真，转向 SWE-Bench Pro 等更难评测。

所以 Codex 强的可能配方是：

```text
强通用推理模型
+ 大规模代码/PR/issue/terminal 数据
+ 真实仓库任务 RL
+ sandbox 中可验证测试反馈
+ PR 偏好/代码审查偏好建模
+ Codex CLI/Cloud harness 的上下文与工具训练
+ 多任务并行、自动验证、回放和安全策略
```

### Anthropic/Claude Code 线索

Anthropic 公开资料更偏工程方法论：

- Claude Code best practices：强调 repo-aware 使用、并行 session、测试和迭代。
- Effective context engineering：把上下文设计视为 agent 能力核心。
- Effective harnesses for long-running agents：强调 multi-context window、initializer、环境管理。
- Writing tools for agents、advanced tool use、MCP code execution：强调工具定义、工具发现、减少 token 浪费。
- Claude Code sandboxing、auto mode：用权限分类器和 filesystem/network isolation 降低确认疲劳，同时防 prompt injection 和数据外泄。
- Harness design for long-running application development、Building a C compiler with parallel Claudes：反复说明测试 harness、反馈质量、多 agent 并行会显著影响结果。

所以 Claude Code 强的可能配方是：

```text
强长上下文/推理模型
+ 大量真实 agentic coding 使用轨迹
+ 极重视工具与上下文格式
+ 多窗口/记忆/压缩/initializer 工作流
+ sandbox + permission classifier
+ 高质量 eval 和 failure-mode-driven harness 改进
+ 并行 agent 编排与人类 review 闭环
```

两家的共同点是：**模型本身和 agent runtime 已经不可分割**。一个 frontier coding model 往往是在特定 harness、特定工具接口、特定 sandbox、特定轨迹格式里训练/评测/迭代出来的。

## 资料雷达：按方向整理的 60+ 篇/页

### A. SWE benchmark 与评测污染

| 资料 | 要点 |
| :--- | :--- |
| [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770) | 真实 GitHub issue/PR + 测试验证，SWE agent 领域起点。 |
| [SWE-bench GitHub](https://github.com/swe-bench/SWE-bench) | 官方代码与数据加载入口。 |
| [SWE-bench 官网/leaderboard](https://www.swebench.com/) | 追踪 SWE-bench、mini-SWE-agent、SWE-smith 等动态。 |
| [OpenAI o1 System Card: SWE-bench Verified](https://openai.com/index/openai-o1-system-card/) | 介绍 Verified 的人工复核背景。 |
| [SWE-Bench+: Enhanced Coding Benchmark for LLMs](https://arxiv.org/abs/2410.06992) | 指出 solution leakage、weak tests、数据污染等问题。 |
| [SWE-Bench Pro](https://arxiv.org/html/2509.16941v1) | 更难、更接近企业级、更多语言/仓库。 |
| [OpenAI: Why SWE-bench Verified no longer measures frontier coding capabilities](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) | 官方讨论 Verified 污染与测量失真。 |
| [OpenAI: Separating signal from noise in coding evaluations](https://openai.com/index/separating-signal-from-noise-coding-evaluations/) | 延续 coding eval 的信号/噪声问题。 |
| [SWE-Lancer](https://openai.com/index/swe-lancer/) | Upwork 真实 freelance 任务，包含工程和管理判断。 |
| [Multi-SWE-bench](https://arxiv.org/abs/2504.02605) | 多语言 issue resolving。 |
| [SWE-bench Multimodal](https://arxiv.org/abs/2410.03859) | 把视觉软件任务纳入 SWE 评测。 |
| [Rust-SWE-bench](https://arxiv.org/html/2602.22764v1) | Rust 生态的仓库级 issue resolving。 |
| [DeepSWE benchmark](https://github.com/datacurve-ai/deep-swe) | 113 个原创长周期软件工程任务，覆盖 TS/Go/Python/JS/Rust。 |
| [Dissecting the SWE-Bench Leaderboards](https://arxiv.org/html/2506.17208v1) | 分析 SWE-bench leaderboard 的提交者与架构。 |
| [A Survey of LLM-based Software Repair](https://arxiv.org/html/2506.23749v2) | LLM 自动修复系统综述。 |

### B. SWE 数据合成、gym 与轨迹训练

| 资料 | 要点 |
| :--- | :--- |
| [SWE-smith](https://arxiv.org/abs/2504.21798) | 任意 Python 仓库生成 100s-1000s 可验证任务，50k+ instances。 |
| [SWE-smith GitHub](https://github.com/SWE-bench/SWE-smith) | 训练 SWE-agents 的工具链。 |
| [SWE-smith 项目页](https://swesmith.com/) | 官方介绍与数据规模。 |
| [SWE-Gym](https://arxiv.org/abs/2412.21139) | 2,438 个真实 Python 任务，训练 agent 和 verifier。 |
| [SWE-Gym Hugging Face](https://huggingface.co/SWE-Gym) | 数据、模型、轨迹入口。 |
| [SWE-Fixer](https://arxiv.org/abs/2501.05040) | 110K GitHub issues + retrieve/edit 两阶段训练。 |
| [SWE-Fixer GitHub](https://github.com/InternLM/SWE-Fixer) | 开源框架。 |
| [SWE-Fixer Editor 72B HF](https://huggingface.co/internlm/SWE-Fixer-Editor-72B) | 编辑模型权重入口。 |
| [SWE-Dev](https://aclanthology.org/2025.findings-acl.193.pdf) | 合成测试和 agent trajectories，训练 + inference scaling。 |
| [R2E-Gym](https://arxiv.org/html/2504.07164v1) | 8K+ 可执行 SWE 环境与 hybrid verifier。 |
| [R2E-Gym 项目页](https://r2e-gym.github.io/) | SWE-Gen 和训练环境介绍。 |
| [Training Long-Context, Multi-Turn SWE Agents with RL](https://arxiv.org/html/2508.03501v1) | 用 DAPO 类 RL 训练 Qwen2.5-72B SWE agent。 |
| [DeepSWE blog](https://www.together.ai/blog/deepswe) | 基于 Qwen3-32B，用 RL 从头训练开源 coding agent。 |
| [DeepSWE-Preview HF](https://huggingface.co/agentica-org/DeepSWE-Preview) | 开源权重与 SWE-bench Verified 结果。 |
| [RepoForge](https://arxiv.org/html/2508.01550v1) | SFT + RL 训练 fast-thinking SWE agent。 |
| [Hybrid-Gym](https://arxiv.org/html/2602.16819v1) | 合成任务泛化到 SWE/SWT/Commit-0 等真实任务。 |
| [SWE-Playground](https://arxiv.org/html/2512.12216) | 从零合成项目、任务、starter code、单测。 |
| [SWE-World](https://huggingface.co/papers/2602.03419) | 用 learned surrogate 替代 Docker 物理环境。 |
| [Scale-SWE](https://arxiv.org/html/2602.09892v4) | 多 agent 沙箱流水线规模化构造 SWE 数据。 |
| [Self-play SWE-RL](https://sophon.at/papers/arxiv-2512.18552) | 用 self-play 迈向少人工依赖的软件 agent 训练。 |
| [Kimi-Dev](https://arxiv.org/html/2509.23045v2) | Agentless training 作为 SWE-agent 的 skill prior。 |

### C. Agent 架构、harness 与开源系统

| 资料 | 要点 |
| :--- | :--- |
| [SWE-agent](https://arxiv.org/abs/2405.15793) | ACI 证明工具接口设计会显著影响 agent 能力。 |
| [SWE-agent GitHub](https://github.com/swe-agent/swe-agent) | 开源 SWE agent 框架。 |
| [Agentless](https://arxiv.org/abs/2407.01489) | 三阶段 localization/repair/validation，提醒复杂 agent 不是唯一基线。 |
| [AutoCodeRover](https://arxiv.org/abs/2404.05427) | 结构化代码搜索 + program improvement。 |
| [OpenHands CodeAct 2.1](https://www.openhands.dev/blog/openhands-codeact-21-an-open-state-of-the-art-software-development-agent) | 开源软件开发 agent，展示 harness 对同底座模型的重要性。 |
| [OpenHands critic model](https://www.openhands.dev/blog/sota-on-swe-bench-verified-with-inference-time-scaling-and-critic-model) | 用 critic/test-time scaling 选择多次尝试中的最佳 patch。 |
| [Nebius: training and search for SWE agents](https://nebius.com/blog/posts/training-and-search-for-software-engineering-agents) | 开源模型 + search 提升 SWE-bench Verified。 |
| [Confucius Code Agent](https://arxiv.org/html/2512.10398v6) | 大仓库、长周期、复杂工具链下的可扩展 agent scaffold。 |

### D. Terminal、CLI 与 computer-use 训练

| 资料 | 要点 |
| :--- | :--- |
| [Terminal-Bench 2.0](https://arxiv.org/abs/2601.11868) | 89 个真实 terminal workflow，强调长周期命令行任务。 |
| [Terminal-Bench GitHub](https://github.com/harbor-framework/terminal-bench) | 终端 benchmark 和 harness。 |
| [Terminal-Bench 官网](https://www.tbench.ai/) | leaderboard 与后续 Terminal-Bench 3.0/Science。 |
| [CLI-Gym](https://arxiv.org/abs/2602.10999) | 环境反演生成 1,655 个 CLI 任务，并训练 LiberCoder。 |
| [Learning CLI Agents with Structured Action Credit](https://arxiv.org/abs/2605.08013) | 用 CLI action 结构做 credit assignment。 |
| [On Data Engineering for Scaling LLM Terminal Capabilities](https://arxiv.org/html/2602.21193v1) | terminal 数据工程和 dataset-to-CLI adapter 思路。 |
| [Terminal-World](https://arxiv.org/abs/2605.20876) | 用 agent skills 合成 terminal-agent environments。 |
| [OSWorld](https://arxiv.org/abs/2404.07972) | 真实电脑环境 benchmark，GUI/文件/多 app 工作流。 |
| [OSWorld 2.0](https://arxiv.org/abs/2606.29537) | 更长周期 computer-use workflow，平均数百 tool calls。 |
| [CUA-Gym](https://arxiv.org/abs/2605.25624) | 可验证 computer-use RLVR 数据合成。 |
| [OSWorld-MCP](https://arxiv.org/html/2510.24563v2) | 评估 computer-use agent 的 MCP 工具调用。 |

### E. 长上下文、仓库级理解与上下文工程

| 资料 | 要点 |
| :--- | :--- |
| [RepoBench](https://arxiv.org/abs/2306.03091) | 仓库级补全、检索、pipeline 任务。 |
| [Long Code Arena](https://arxiv.org/abs/2406.11612) | 六类项目级长上下文代码任务。 |
| [Long Code Arena HF Collection](https://huggingface.co/collections/JetBrains-Research/long-code-arena) | 数据集集合。 |
| [RepoQA](https://arxiv.org/abs/2406.06025) | 长上下文代码理解，500 个函数搜索任务。 |
| [RepoQA GitHub](https://github.com/evalplus/repoqa) | 数据和 benchmark。 |
| [LongCodeU](https://arxiv.org/html/2503.04359v1) | 长代码理解的八类任务。 |
| [ExecRepoBench](https://arxiv.org/abs/2412.11990) | 可执行仓库级代码补全和 Repo-Instruct。 |
| [Coding Agents are Effective Long-Context Processors](https://arxiv.org/abs/2603.20432) | 把长上下文处理外部化到文件系统和可执行工具。 |
| [NL2Repo-Bench](https://arxiv.org/html/2512.12730v2) | 从自然语言需求生成完整 Python 仓库。 |
| [RepoReason](https://arxiv.org/pdf/2601.03731) | 仓库级 white-box diagnostic reasoning benchmark。 |
| [RepoGraph ICLR slides](https://iclr.cc/media/iclr-2025/Slides/28957.pdf) | 用 repository-level code graph 增强 SWE agent 上下文。 |

### F. 工具调用、MCP 与 agent tool learning

| 资料 | 要点 |
| :--- | :--- |
| [ToolLLM / ToolBench](https://arxiv.org/abs/2307.16789) | 16K+ real-world APIs，自动构造 tool-use 数据。 |
| [ToolLLM GitHub](https://github.com/beijixiong1/ToolLLM) | ToolBench 数据与训练代码。 |
| [ToolACE](https://arxiv.org/html/2409.00920v2) | 自动生成准确、复杂、多样的 tool-learning 数据。 |
| [ToolACE Dataset HF](https://huggingface.co/datasets/Team-ACE/ToolACE) | 26,507 APIs，multi-agent 对话生成与验证。 |
| [ToolACE-R](https://arxiv.org/html/2504.01400v3) | model-aware iterative training + adaptive refinement。 |
| [BFCL V4](https://gorilla.cs.berkeley.edu/leaderboard.html) | function calling/tool use leaderboard。 |
| [BFCL paper](https://proceedings.mlr.press/v267/patil25a.html) | AST evaluator、并行/串行函数调用、多步 stateful setting。 |
| [MCP-Universe](https://arxiv.org/abs/2508.14704) | 真实 MCP servers 上的 hard agent benchmark。 |
| [MCP-Universe GitHub](https://github.com/SalesforceAIResearch/MCP-Universe) | RL training、benchmarking、agent development framework。 |

### G. 基础 code model 与 repo-level code training

| 资料 | 要点 |
| :--- | :--- |
| [Qwen2.5-Coder Technical Report](https://arxiv.org/html/2409.12186v1) | 5.5T code-related tokens，source/text-code/synthetic/math/text。 |
| [Qwen2.5-Coder blog](https://qwenlm.github.io/blog/qwen2.5-coder/) | 官方介绍 code data scaling 和能力提升。 |
| [DeepSeek-Coder-V2](https://arxiv.org/html/2406.11931v1) | MoE code model，额外 6T tokens，60% code/10% math/30% NL。 |
| [DeepSeek-Coder-V2 GitHub](https://github.com/deepseek-ai/deepSeek-Coder-V2) | 模型与使用入口。 |
| [StarCoder2 and The Stack v2](https://arxiv.org/abs/2402.19173) | Software Heritage + GitHub PR + notebooks + docs，3.3-4.3T tokens。 |
| [StarCoder2 HF blog](https://huggingface.co/blog/starcoder2) | 模型、论文、使用说明。 |
| [StarCoder2 GitHub](https://github.com/bigcode-project/starcoder2) | 3B/7B/15B，600+ 语言。 |

### H. OpenAI Codex 官方线索

| 资料 | 要点 |
| :--- | :--- |
| [Introducing Codex](https://openai.com/index/introducing-codex/) | 云端软件工程 agent，parallel tasks，独立 sandbox。 |
| [o3/o4-mini Codex system card addendum](https://openai.com/index/o3-o4-mini-codex-system-card-addendum/) | cloud container、无互联网、trajectory 开始方式。 |
| [GPT-5-Codex system card addendum](https://openai.com/index/gpt-5-system-card-addendum-gpt-5-codex/) | 明确提到真实 coding tasks 的 RL 和测试迭代。 |
| [Introducing GPT-5.2-Codex](https://openai.com/index/introducing-gpt-5-2-codex/) | agentic coding、context compaction、Windows、cyber。 |
| [Unrolling the Codex agent loop](https://openai.com/index/unrolling-the-codex-agent-loop/) | Codex CLI agent loop、tool schema、context 管理。 |
| [Unlocking the Codex harness](https://openai.com/index/unlocking-the-codex-harness/) | App Server、Codex surfaces、harness 集成。 |
| [Harness engineering with Codex](https://openai.com/index/harness-engineering/) | agent-first 工程组织与 repo knowledge。 |
| [Open-source Codex orchestration: Symphony](https://openai.com/index/open-source-codex-orchestration-symphony/) | 多 Codex session 编排与上下文切换成本。 |
| [Introducing the Codex app](https://openai.com/index/introducing-the-codex-app/) | 从单 agent pair 到 agent team 生命周期。 |

### I. Anthropic / Claude Code 官方线索

| 资料 | 要点 |
| :--- | :--- |
| [Claude Code best practices](https://www.anthropic.com/engineering/claude-code-best-practices) | 配置环境、并行 session、agentic coding 模式。 |
| [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) | workflow 和 agent 的基础设计模式。 |
| [Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) | 上下文工程是 agent 可控性的核心。 |
| [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) | initializer、多 context window、环境管理。 |
| [Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps) | frontend/长周期应用开发中的 harness 设计。 |
| [Writing tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) | 工具质量、工具评测、让 agent 优化工具。 |
| [Code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp) | 用 code execution 降低 MCP 工具定义和结果的 token 成本。 |
| [Advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use) | on-demand tool discovery，避免把所有工具塞进上下文。 |
| [Claude think tool](https://www.anthropic.com/engineering/claude-think-tool) | 复杂工具使用中给模型显式思考空间。 |
| [Claude Code sandboxing](https://www.anthropic.com/engineering/claude-code-sandboxing) | filesystem/network isolation，降低 prompt injection 风险。 |
| [Claude Code auto mode](https://www.anthropic.com/engineering/claude-code-auto-mode) | permission classifier，减少确认疲劳。 |
| [Building a C compiler with parallel Claudes](https://www.anthropic.com/engineering/building-c-compiler) | 多 agent 并行、测试 harness、反馈质量。 |
| [Enabling Claude Code to work more autonomously](https://www.anthropic.com/news/enabling-claude-code-to-work-more-autonomously) | Claude Code 2.0、IDE、checkpoint、更长任务。 |

## 我会优先读的 12 篇

如果只想快速建立“coding agent 为什么强”的技术直觉，我会按这个顺序读：

1. [SWE-bench](https://arxiv.org/abs/2310.06770)
2. [SWE-agent](https://arxiv.org/abs/2405.15793)
3. [Agentless](https://arxiv.org/abs/2407.01489)
4. [SWE-Gym](https://arxiv.org/abs/2412.21139)
5. [SWE-smith](https://arxiv.org/abs/2504.21798)
6. [R2E-Gym](https://arxiv.org/html/2504.07164v1)
7. [Training Long-Context, Multi-Turn SWE Agents with RL](https://arxiv.org/html/2508.03501v1)
8. [Terminal-Bench](https://arxiv.org/abs/2601.11868)
9. [CLI-Gym](https://arxiv.org/abs/2602.10999)
10. [Coding Agents are Effective Long-Context Processors](https://arxiv.org/abs/2603.20432)
11. [Codex agent loop](https://openai.com/index/unrolling-the-codex-agent-loop/)
12. [Anthropic context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

读完这 12 篇，基本就能把主线串起来：**SWE-bench 定义问题，SWE-agent/OpenHands 定义 agent interface，SWE-Gym/SWE-smith/R2E-Gym/CLI-Gym 解决数据和环境，RL/verifier/test-time scaling 解决训练，Codex/Claude 的官方文章展示工业 harness 如何把模型能力变成产品能力。**

## 对“怎么把 coding agent 做牛”的实践判断

如果要自己训练或搭一个强 coding agent，我会把资源优先级排成这样：

1. **先做评测和环境，不要先改模型**：没有稳定可复现的 sandbox/test/verifier，训练出来也不知道是真进步还是 benchmark hack。
2. **从真实仓库任务种子开始**：issue/PR/test/log 比纯合成函数题更接近真实能力。
3. **把 terminal trajectory 当一等数据**：命令、输出、失败日志、下一步动作，全都要结构化存下来。
4. **训练 retrieve/edit/test 的行为先验**：先用 SFT 学会像开发者一样工作，再用 RL 优化最终通过率。
5. **训练 verifier/critic，而不是只训练 actor**：多尝试 + 会挑 patch，往往比单次生成更有效。
6. **让 harness 可观测、可回放、可控权限**：agent 能力越强，越需要 sandbox、日志、权限、diff、回滚。
7. **长上下文要靠工程，不只靠窗口**：repo map、摘要、记忆、压缩、工具发现、context budget 都要进入系统设计。
8. **持续做失败模式分析**：前沿差距往往来自失败样本：定位失败、误读 issue、过拟合测试、环境错误、忘验证。

最后一句话总结：**GPT/Codex 和 Claude Code 变强，靠的是“模型在真实软件工程环境里被训练成开发者”，而不是“聊天模型突然更会写代码”。** 这里的“开发者”不是人设，而是一组可训练、可验证、可回放的行为轨迹：读仓库、找上下文、跑命令、看日志、改 patch、跑测试、接受反馈、继续迭代。

