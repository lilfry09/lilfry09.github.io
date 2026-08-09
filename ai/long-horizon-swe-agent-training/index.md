# 训练模型的长程软件工程能力：从 SWE-Marathon 到 FrontierSWE 与 SWE-Together


如果要训练模型做长程软件工程任务，最容易走错的一步，是把“长程能力”理解成“更长上下文 + 更多 token”。SWE-Marathon、FrontierSWE 和 SWE-Together 这三组工作放在一起看，会给出一个更锋利的答案：长程能力不是文本长度，而是 **在一个可执行环境里，持续维护目标、状态、证据、风险和验证闭环**。

SWE-bench 把真实 GitHub issue 变成了可评分 patch，这是很重要的一步。但它主要回答“模型能不能修一个相对局部的问题”。现在这三组新 benchmark 关心的是另一件事：模型能不能像一个还算可靠的工程师那样，连续几个小时推进一个项目，把中途实验、失败、回滚、测试、用户纠偏、最终提交都组织起来。

<!--more-->

## 如果只记住三句话

第一，SWE-Marathon 测的是“能不能把一个完整项目做完”：产品克隆、库重写、ML 训练、系统优化，全都放进 Harbor/Docker 环境，用 hidden verifier 判定最终是否 resolved。

第二，FrontierSWE 测的是“能不能在人类专家边界上做出真实进展”：PostgreSQL wire-compatible server、Pyright 性能优化、optimizer research 这类任务不再适合二值 resolved，需要 correctness gate、speedup、held-out metric 和 anti-cheat。

第三，SWE-Together 补上了过去 benchmark 缺的一块：真实用户不是一次性把需求说完，而会追问、纠偏、追加约束；因此 trajectory 不能只存最终 patch，还要存每一次用户意图更新、模型行动、工具结果和 verifier 反馈。

## 1. 三个 benchmark 分别在测什么

先把边界理清楚。它们名字都带 SWE，但关注点其实不同。

| Benchmark | Canonical 形态 | 任务数量 | 核心问题 | 最重要的训练信号 |
| :--- | :--- | :--- | :--- | :--- |
| SWE-Marathon | arXiv 2606.07682 + GitHub + 官网 leaderboard | 论文 v1 为 20 个任务 | agent 能否自主完成 ultra-long-horizon software work | 项目级施工、长时间验证、失败恢复、最终 resolved |
| FrontierSWE | Proximal 官方博客 + GitHub + leaderboard；不是 arXiv 论文 | 首发 17 个任务 | coding agent 在专家级系统/性能/研究挑战上能推进到哪里 | continuous reward、correctness-gated speedup、best-so-far 管理 |
| SWE-Together | arXiv 2606.29957 + GitHub + Hugging Face dataset | 109 个任务 | agent 能否在多轮真实用户会话中完成任务 | 用户纠偏、追加需求、交互成本、intent coverage |

这里要特别小心版本漂移。SWE-Marathon 的论文数字、官网 leaderboard、GitHub main 和 Hugging Face 快照已经有差异；FrontierSWE 当前也没有论文，应该引用 Proximal blog 和仓库；SWE-Together 的可运行任务目录和 Hugging Face 表格互相映射，但真正能跑的是 `tasks/<task_id>/` 里的 Dockerfile、verifier 和 prompt。

这不是吹毛求疵。做训练数据时，`task_id` 还不够，必须把 `task_version_id`、repo commit、Docker image digest、harness commit、verifier 版本一起存下来。否则两条轨迹看起来来自同一个任务，实际评分逻辑已经变了。

## 2. 作者动机：为什么 SWE-bench 不够了

SWE-Marathon 的问题意识最直接：现有 coding benchmark 多是短 horizon。SWE-bench 很好，因为它来自真实 issue/PR，但它依然偏向“定位一个 bug，改一个 patch，跑测试”。真正的软件工程任务经常不是这样。你要先读系统、补 scaffold、做实验、修一堆边角、反复验证，最后还要知道什么时候提交。

论文里一个很有意思的结果是，长程 rollout 的 token 绝大多数不是模型新生成的答案，而是系统提示、历史工具输出和上下文回放。换句话说，成本烧在“保持工作状态”上，而不是烧在“想出一句神奇代码”上。这解释了为什么最高 token 使用量并不必然带来最高通过率：长程任务的瓶颈不是单次推理长度，而是工作循环质量。

FrontierSWE 的动机更像 capability probe。Proximal 的官方博客把矛头指向现有 benchmark 的任务尺度和评分方式：当 agent 已经开始做大规模 refactor、性能工程、安全研究、甚至从零搭系统时，用短任务和二值 resolved 很难区分 frontier model 之间的差异。它的任务数量更少，但每题都很重：implementation、performance、ML research 三类任务都要求专家级背景。

SWE-Together 则从另一个角度补洞：真实用户会话不是一条 instruction 到底。用户会问“为什么这样改”、会说“不是改依赖，是在这个 repo 里修”、会提醒“别 mock 太多，测真实路径”、会追加一个性能分析问题。传统 benchmark 把这些互动压扁成一个最终 patch，训练出来的 agent 就容易只会“闭门造车”，不会根据用户反馈改变轨迹。

所以三者合起来的核心动机是：

```text
SWE-bench:   真实 issue → 可验证 patch
SWE-Marathon: 完整项目 → 长程施工与最终验收
FrontierSWE: 专家挑战 → 连续指标与高成本实验
SWE-Together: 真实会话 → 用户意图随时间更新
```

## 3. 任务类型与真实例子

下面这些例子都来自任务文件、verifier 或 reference patch，而不是只读 abstract。

### 3.1 SWE-Marathon：完整项目级任务

`slack-clone` 是一个很典型的“长程产品 + 分布式系统 + UI”任务。任务要求在 `/app` 里实现一个 Slack-like team chat system：三个 HTTP 节点分别监听 `127.0.0.1:8000/8001/8002`，还有一个 IRC gateway 在 `:6667`，并提供浏览器 UI。功能面包括注册登录、workspace、public/private channel、DM、message、thread、reaction、pin、file、search、user group、slash command、mention、read state 和 settings。

难点不只是功能多。所有 HTTP 节点和 IRC gateway 必须共享 per-channel dense sequence；WebSocket reconnect 到另一个节点后要能 replay；杀掉一个 HTTP node 后其他节点继续读写和推送；杀掉 cross-node broadcast component 后，系统还要在 5 秒内通过 SQLite fallback 保持 fan-out 和 dense seq。前端还要过 Playwright 和 Computer-Use UI rubric。当前 `task.toml` 里专家估时是 60 小时，agent 时间约 3 小时。

这个任务训练的不是“写一个聊天页面”，而是：

- 如何先搭可启动 skeleton，而不是沉迷局部模型；
- 如何把 API、持久化、实时推送、IRC、UI 分层推进；
- 如何在 chaos test 下维护一致性；
- 如何避免只通过 visible browser test，却在 hidden API/cluster test 上崩掉。

`biofabric-rust-rewrite` 是另一种长程：把 Java BioFabric 和 Network Alignment plugin 重写成 Rust library + CLI。skeleton 会编译，但大量函数是 `todo!()`；public API 不能乱改；可见和隐藏 parity tests 会比较 BIF/NOA/EDA、layout、analysis、alignment 等输出，要求接近 byte-level parity。任务 metadata 里专家估时 80 小时。

这类任务对训练 trajectory 很有价值，因为它逼 agent 学会“参考实现迁移”的纪律：先找 Java 行为真相，锁定 public API，逐模块实现，跑小 parity，再跑大 parity。成功不是一次 patch，而是很多微小语义对齐的累积。

`parameter-golf` 则提醒我们，SWE-Marathon 不是只有传统代码工程。它要求在 WikiText BPE-8K 数据上训练一个语言模型，最终提交 `/workspace/train_gpt.py` 和不超过 32MB 的压缩 checkpoint，held-out `val_bpb < 0.983` 才算过。任务涉及模型结构、训练 schedule、量化压缩、H100 时间预算和反作弊检查。它训练的是长程 ML engineering：实验设计、资源预算、checkpoint 管理和 final artifact contract。

### 3.2 FrontierSWE：专家级系统、性能与研究挑战

FrontierSWE 的 `postgres-sqlite-wire-adapter` 要求写一个 Zig 程序，用 SQLite 做底层存储，却在 PostgreSQL 18.3 客户端看来像真实 PostgreSQL server。单个二进制需要根据 `argv[0]` 扮演 `postgres`、`initdb`、`pg_ctl`。隐藏 verifier 会把它接进 PostgreSQL 官方 regression suite 和 TAP tests，还包括 startup packet、auth、simple query、prepared statement、transaction、catalog/introspection、Unix socket、CLI behavior、cluster lifecycle 等兼容面。

当前 verifier 里可以看到一个很硬的评分形态：source scan、Zig project、dependency、build、binary、hidden harness 任一硬失败都置零；如果能跑起来，则按 regression、TAP、compatibility tests 的整体通过率给分。这个任务几乎是“用 SQLite 写一个 PostgreSQL 影子”。它训练的不是单个 SQL feature，而是如何优先实现 public compatibility surface，尽早让 `psql select 1` 和 `pg_ctl start/stop` 跑通，再逐步扩大 SQL/catalog 覆盖。

`revideo-perf-opt` 是性能工程任务。Revideo 是 TypeScript 视频生成框架，渲染链路包括 Puppeteer/Chrome、Vite、Canvas、WebSocket、FFmpeg。instruction 明确提示可能瓶颈：`canvas.toBlob()`、browser-node round trip、FFmpeg WebSocket bridge、视频 decode/encode 路径。agent 可以改任意 package、用 WebCodecs/WASM/muxer，但不能改变视觉输出、跳帧、降质或改 API。

verifier 先重建 candidate，再用 hidden scenes 跑 baseline 和 candidate，比较 SSIM/时长等 correctness，只有所有 hidden scenes 正确，才把几何平均 speedup 作为 reward。这个 scoring 特别关键：性能类 trajectory 不能只记录“快了”，必须记录 correctness gate 是否仍然全过。否则模型会学会危险的“快但错”。

`pcqm4mv2-autoresearch` 是 ML research 任务。agent 要在 closed-data、2D-only、no conformer、no external checkpoint 的约束下做 molecular graph regression，参数上限 50M，最终只允许 `/app/submission/` 和 `/app/checkpoint/` 里的内容参与 verifier-time inference。hidden test 与 visible dev 分离，要求 `predict.py --count-params`、`--input-path`、`--output-path` contract 都正确。

它训练的是“可执行研究”而非论文摘要：看数据 schema、设计 pipeline、跑实验、读 dev MAE、保留最好 checkpoint、控制参数量、保证最终提交最小且可加载。很多 agent 的失败不在模型结构，而在最后 artifact contract：checkpoint 放错、predict 依赖散落、hidden 输入假设错。

### 3.3 SWE-Together：真实多轮用户会话

SWE-Together 的一个短 bugfix 例子是 `comfyui-triton-windows-amd-fix`。初始用户消息是 Windows + AMD GPU 下 Triton 编译报错，指向 `ComfyUI-WanVideoWrapper/ultravico/sageattn/attn_qk_int8_per_block.py`。reference patch 很小：把循环里的 `tl.load(K_scale_ptr)` 改成基于稳定 base pointer 的 indexed load，并删除 `K_scale_ptr += 1`。但任务不是“猜这两行”。用户模拟器会在 agent 没解释编辑时问“Why is this modification needed?”，在 agent 偏去修 Triton C++ 后问“如何在这个 repo 里修，而不是改 Triton kernel？”

这说明 SWE-Together 的任务粒度不是 patch LOC，而是交互行为：agent 要懂用户在限制 scope、要求解释、纠正错误方向。canonical goals 里也把“不要改外部依赖、不要 silent edit、要解释原因”列为评分目标，而不仅是代码 shape。

另一个更长例子是 `qwen3-moe-gguf-dequant`。初始要求是修 `dequantize_blocks_IQ3_XXS`，并参考 llama.cpp 的 `gguf-py/gguf/quants.py`。后续任务实际上扩展到实现多种 IQ dequant：`IQ3_S`、`IQ1_S`、`IQ2_S`、`IQ2_XXS`、`IQ1_M` 等，要求用 PyTorch elemental operations 对齐 reference。用户模拟器的条件包括：如果 agent 过度折腾 `PYTHONPATH`，提醒“直接在当前目录跑 `python test_gguf_dequant.py`”；如果还用 `F.embedding`，要求换成 elemental torch ops；如果 `IQ1_S` 性能看起来慢，追问瓶颈是不是 dequant 代码。

这类 trajectory 对训练尤其有用，因为它展示了长程会话里的细粒度能力：测试协议服从、实现多个相似但不相同的 bit layout、解释性能现象、判断瓶颈在 test-side quantization 还是自己代码。

第三个例子 `pi-mono-parallel-tool-stall` 更接近 harness/debug 任务。初始要求是调查 parallel tool execution 和 interactive tools stall，不要实现，只产出 memo。但真实会话逐步推进到“先加 failing test”“不要 interactive metadata”“不要 throw，让它们 sequential”“测试要用真实代码路径，不要 mock/recreate 太多”“两个 submission 都要完成或至少尝试”。reference patch 最终是在 SDK 创建 session 时设置 `toolExecution: "sequential"`，并加了一个真实 session wiring 的 Vitest。

这个例子很漂亮地说明：长程软件工程任务的目标会在对话中变化。turn 0 说“不要实现”，但用户后面明确要求写 failing test 和修复；如果 harness 只保存初始 instruction，不保存用户纠偏，就会误判正确行为。

## 4. 任务是如何构造的

三个 benchmark 的构造方式可以抽象成同一套“可执行任务资产”：

```text
instruction.md
task metadata: timeout / CPU / GPU / memory / network / image
environment: Dockerfile / pinned repo commit / mounted data
visible feedback: smoke tests / dev benchmark / public examples
hidden verifier: tests/test.sh / compute_reward.py / judge rubric
reference patch or oracle
anti-cheat: source scan / dependency scan / hash / hidden asset isolation
trajectory capture: agent logs / tool calls / verifier metrics / final workspace
```

SWE-Marathon 是人工 authored project tasks。贡献者需要给 objective、Docker、visible checks、hidden verifier、oracle、专家估时、资源需求和 reward-hack 风险。任务要通过 proposal review、CI、oracle run、NOP run、frontier-agent pilot、人工读日志、adversarial cheating agent 等环节。它的 `CONTRIBUTING.md` 把目录固定为 `tasks/<slug>/instruction.md`、`task.toml`、`environment/Dockerfile`、`solution/solve.sh`、`tests/test.sh`。

FrontierSWE 官方披露没有那么像论文 pipeline，但仓库结构能看出实际做法：每个任务有 `instruction.md`、`task.toml`、Docker image、hidden verifier、`compute_reward.py`、可选 `solution/` 和 `oracle.yaml`。评分不是统一二值，而是按任务族定义：implementation 看测试通过率，performance 先过 correctness 再算 speedup，ML research 看 held-out metric。

SWE-Together 的构造则围绕真实 user-agent session。每行数据包含 instruction、repo、base commit、scoring targets、reference patch、user intents、docker image、task id。每个 `tasks/<task_id>/` 目录里有 instruction、user simulation prompt、Dockerfile、tests/verifier、reference patch 和 frozen judge rubric。它的关键创新是 progressive reveal：agent 先得到 turn 0，之后 user simulator 根据原始会话里的触发条件发问、纠偏或保持沉默。

如果我们自己要实践，我会按这个顺序做：

1. 先选真实任务，不要先写 benchmark。来源可以是内部 backlog、性能瓶颈、迁移需求、真实用户 debug 会话、研究实验目标。
2. 固定 repo commit、数据版本、Docker digest、工具链版本和网络策略。
3. 写 outcome-oriented instruction，只告诉 agent 目标、约束、接口和 visible feedback，不泄漏 hidden verifier。
4. 做最小 visible smoke test，让 agent 能确认方向，但不要让 visible test 等于 final hidden set。
5. 先写 oracle/reference solution，再跑 NOP baseline。oracle 必须过，NOP 必须低分。
6. 设计 task-specific reward，而不是偷懒统一 pass/fail。性能任务要 correctness gate；ML 任务要 held-out；产品任务要 API + UI + resilience。
7. 加 anti-cheat：隐藏资产结束后才挂载；source/dependency scan；禁止读 `/tests`、`/baseline`；hash 检查；外网 allowlist。
8. 跑 3-5 个 frontier-agent pilot，人工读失败日志，区分“任务歧义”与“模型能力失败”。
9. 冻结 task version，把所有 rollout 都绑定 `task_hash + verifier_hash + harness_hash`。
10. 最后才大规模采 trajectory。

这里的一个原则是：**先让任务成为可靠评测，再让它成为训练数据**。如果 oracle 不能稳定通过，或者 NOP 也有高分，这个任务不适合训练。

## 5. Trajectory 到底应该保存什么

长程 trajectory 不是聊天记录。真正有训练价值的是状态转移：

```json
{
  "task_id": "revideo-perf-opt",
  "task_version": "...",
  "harness": "claude-code",
  "state": {
    "changed_files": ["packages/core/src/exporter/..."],
    "best_score_so_far": 1.42,
    "known_failures": ["hidden correctness unknown"],
    "remaining_budget_sec": 18300
  },
  "model_action": {
    "kind": "run_benchmark",
    "command": "node benchmark.mjs"
  },
  "observation": {
    "correctness": "public_pass",
    "speedup": 1.18,
    "stderr_tail": "..."
  },
  "decision_label": {
    "outcome": "keep_experiment",
    "reason": "improved public speed without visible parity regression"
  }
}
```

最低限度要保存这些字段：

| 类别 | 必须保存 |
| :--- | :--- |
| 任务身份 | benchmark、task id、task version、repo commit、image digest、verifier hash |
| harness 身份 | agent scaffold、tool definitions、permission mode、system prompt、skills/MCP |
| 用户侧 | 初始 instruction、后续 user messages、message tags、correction/nudge |
| 模型侧 | assistant text、tool call、arguments、reasoning/rationale 是否可用 |
| 工具侧 | stdout/stderr、exit code、timeout、文件 diff、测试结果 |
| 状态侧 | changed files、checkpoint、best-so-far、remaining budget、open blockers |
| verifier 侧 | final reward、partial score、subscores、hard fail reason、anti-cheat |
| 成本侧 | input/cache/output tokens、cost、wall time、number of turns |

SWE-Marathon 的全量 raw logs README 说约 320GB，但目前需要向作者申请 S3 credentials；官网压缩 trajectory JSON 可以匿名看，不过 schema 更像展示用摘要，常见结构是 `trial` 加 `rows[{step, call, kind, title, detail}]`，不是完整可 replay 的 ATIF。FrontierSWE 当前公开仓库没有 raw trial dataset，但 Claude Code adapter 会把 `sessions/**/*.jsonl` 转成 `/logs/agent/trajectory.json`。SWE-Together 则把可运行任务、用户模拟 prompt、reference patch 和 frozen goals 公开得更完整。

一个重要判断：**不要把“最终通过”直接当整条轨迹正样本**。长程轨迹里有很多失败动作、误读、重复、reward hacking、自动重试、harness 托举。成功 trial 里也可能有很多不该学的片段；失败 trial 里也可能有非常好的局部诊断。蒸馏应该切 episode，而不是整条吞。

## 6. Trajectory 怎么蒸馏

我会把蒸馏拆成四种数据，而不是只做一种 SFT。

第一种是 action SFT。输入是 task + 当前 durable state + 最近 observation，目标是下一步高质量行动。例如：

```text
状态：Pyright public diagnostics parity 全过，但 hidden 未知；ABBA timing 显示 union stress 慢。
好动作：保存当前 checkpoint，profile union narrowing path，做小范围缓存实验。
坏动作：直接提交，或重构整个 type evaluator。
```

第二种是 state compression。长程 agent 最容易死在上下文污染和记忆漂移。要训练模型把几小时轨迹压成结构化状态：

```json
{
  "goal": "implement PostgreSQL wire-compatible adapter on SQLite",
  "working": ["initdb creates cluster dir", "TCP startup/auth works", "select 1 works"],
  "failing": ["prepared statement Describe returns wrong RowDescription"],
  "changed_files": ["src/main.zig", "src/protocol.zig"],
  "next": "add minimal Parse/Bind/Describe/Execute support and run psql prepared query smoke"
}
```

第三种是 verifier critic。很多训练样本应该不是“下一步代码”，而是“解释失败并选择验证”。例如 Revideo 任务里，speedup 提高但 SSIM fail，critic 要学会这是不可接受的，不是“部分成功”；Slack clone 里 WebSocket fan-out pass 但 Redis kill 后 seq gap，critic 要定位到 fallback path，而不是继续美化 UI。

第四种是 stop/continue policy。长程任务最贵的错误之一是过早提交或反复提交。SWE-Marathon 的失败模式里有 premature termination，官网失败轨迹也能看到连续 Submit 后又返工。训练时应该构造偏好：

```text
完整 gate 通过 + hidden-like smoke 稳定 + best checkpoint 已恢复 > 自称完成但未跑完整验证
中途最佳 checkpoint > 最终冒险改坏版本
明确回滚 > 在失败实验上继续堆补丁
```

训练配方上可以这样分：

| 数据 | 用途 |
| :--- | :--- |
| 成功轨迹中的高质量局部 episode | SFT / tool-use imitation |
| 同一状态下好坏分支 | DPO / preference |
| verifier subscore 和 hard fail reason | process reward / critic |
| fresh internal rollout | online RL |
| 失败轨迹 | stop policy、failure classifier、anti-pattern mining |

还有几个过滤原则：

- oracle/`solve.sh` 不能当普通 agent 解题数据；
- anti-cheat 失败、verifier tampering、环境异常要剔除或单独标注；
- `is_copied_context`、自动补跑、harness 自动恢复不能当模型功劳；
- tool definitions 和 harness 行为要随样本保存，否则模型学到的是某个框架习惯；
- 公开 benchmark 的 trajectory 不能训练后又回到同一 benchmark 报无污染结果。

## 7. 怎么 hack 进 Claude Code / harness

这里有两条路：不换模型，只增强 harness；或者把自训模型接进 Claude Code。

### 7.1 不换模型：把长程策略做成 harness 能力

这条最稳。你可以把蒸馏出的通用策略写进 `CLAUDE.md`、skill、MCP 或 wrapper：

- 每个任务创建 `acceptance_ledger.md`：目标、约束、visible tests、hidden-like risks；
- 每次重要编辑前保存 checkpoint；
- 每次实验写假设、命令、结果、是否保留；
- 自动维护 best-so-far，而不是默认提交 current workspace；
- 提交前强制跑 full gate 或 task-specific final checklist；
- 剩余时间低于阈值时停止大重构，恢复 best checkpoint；
- 对 repeated failures 触发“换诊断路径”，不是继续同一 patch。

这种方式训练的是 agent system，而不是裸模型。它的好处是马上可用；坏处是训练信号会被 harness 污染，所以采 trajectory 时必须标清“这一步是模型自己决定的，还是 wrapper 自动做的”。

### 7.2 换模型：让 Claude Code 调 Anthropic-compatible student

Claude Code 本身期待 Anthropic Messages / streaming / tool use 语义。FrontierSWE 和 SWE-Marathon 的 Harbor adapter 已经展示了可行路径：用 `ANTHROPIC_BASE_URL`、`ANTHROPIC_MODEL`、`ANTHROPIC_DEFAULT_*_MODEL`、`CLAUDE_CODE_SUBAGENT_MODEL` 把 Claude Code 指到 DeepSeek、GLM 或其他 Anthropic-compatible endpoint。

实践上大概是：

```bash
export ANTHROPIC_BASE_URL="https://your-anthropic-compatible-gateway"
export ANTHROPIC_API_KEY="..."
export ANTHROPIC_MODEL="your-long-horizon-student"
export ANTHROPIC_DEFAULT_OPUS_MODEL="your-long-horizon-student"
export ANTHROPIC_DEFAULT_SONNET_MODEL="your-long-horizon-student"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="your-long-horizon-student"
export CLAUDE_CODE_SUBAGENT_MODEL="your-long-horizon-student"
```

然后用 Claude Code/Harbor 正常跑任务，保留 `--output-format=stream-json` 和 session JSONL，再转成 ATIF trajectory。关键前提是你的 gateway 真正兼容 Claude Code 所需的 streaming、tool call、error、long-session 行为。只提供 OpenAI `/chat/completions` 的 vLLM endpoint 通常不够，需要一个 Anthropic translation layer。

如果目标是严肃训练自有模型，而不是借 Claude Code UI，那么更干净的方案是写自定义 Harbor agent：直接调用你的模型，暴露同样的 bash/edit/read/test 工具，输出 ATIF，顺便保存 token ids、mask ids、logprobs。这样才能做真正 on-policy RL。Claude Code 历史轨迹更适合 SFT/preference bootstrapping，不适合直接当 policy-gradient rollout。

### 7.3 Harness 接入时最该记录的事件

我会在 harness 里加一层语义事件，而不是只存原始 stdout：

```json
{
  "event": "verification_failed",
  "scope": "correctness_gate",
  "command": "bash tests/test.sh",
  "component": "redis_fallback_seq_dense",
  "model_next_action": "inspect channel_events poller",
  "outcome": "recovered_after_two_edits"
}
```

特别要把以下事件标准化：

- `checkpoint_saved`
- `best_checkpoint_updated`
- `experiment_regressed`
- `rollback_performed`
- `verification_skipped`
- `hidden_like_risk_identified`
- `user_correction_received`
- `scope_violation_detected`
- `harness_auto_retry`
- `final_submit`

这些事件才是长程训练里的“骨架”。如果没有它们，trajectory 就只是一大坨 bash 和 diff。

## 8. 评测和反作弊：别把训练集做成答案库

长程 SWE 数据最危险的地方，是它天然包含答案、测试、hidden behavior 和模型探测 verifier 的痕迹。SWE-Marathon 的 HF 卡就提醒过 decontamination；SWE-Together 的 user simulator 也明确要求不能泄漏 fix 变量名、代码模式或行号。

一个可接受的训练/评测隔离应该至少包括：

1. 训练任务池和评测任务池按 repo、task family、verifier asset 严格分开；
2. 公开 benchmark 的 solution/trajectory 只能用于方法研究，不用于同 benchmark 提分后报告；
3. hidden tests 永不进入 prompt、SFT target 或 retrieval memory；
4. 任务文件里加 canary，并在训练语料里扫描；
5. 训练数据保留许可证和来源；
6. 任何用过 public trajectory 的模型，评测时标注 contamination risk；
7. agent 运行期禁用搜索答案、git checkout known solution、读取 `/tests` 或 `/baseline`；
8. verifier 记录 source scan、dependency scan、hash、network、file read traces。

尤其是 performance 和 ML research 任务，reward hacking 很自然：跳帧、降质、偷看 labels、缓存 hidden 输入、改 benchmark、伪造 loss。任务构造时要先设计作弊方案，再设计 anti-cheat，不要等 leaderboard 出现异常再补洞。

## 9. 结论：长程能力训练的真正对象

这三组 benchmark 共同说明了一件事：训练模型做长程软件工程，不是把 SWE-bench patch 变长，也不是把 context window 拉到几百万 token。真正要训练的是一组工程习惯：

- 读清 acceptance criteria；
- 建立可运行 baseline；
- 小步实现；
- 用测试和 benchmark 反馈更新状态；
- 保存 best checkpoint；
- 能解释失败；
- 能听用户纠偏；
- 知道何时回滚、何时继续、何时提交。

SWE-Marathon 给了完整项目施工的舞台；FrontierSWE 把舞台推到专家级系统和研究任务；SWE-Together 把真实用户重新放回 loop 里。它们的共同启发是：未来有价值的 Agent 数据，不会只是“prompt → patch”，而会是：

```text
可执行任务 + 真实环境 + 分阶段反馈 + 用户纠偏 + 状态快照 + 验证指标 + 可归因轨迹
```

训练长程能力，归根到底是在训练模型成为一个可靠的工程过程参与者。模型当然要会写代码，但更要会维护一个还没结束的现实。

## 参考链接

- [SWE-Marathon paper: arXiv 2606.07682](https://arxiv.org/abs/2606.07682)
- [SWE-Marathon GitHub](https://github.com/abundant-ai/swe-marathon)
- [SWE-Marathon website and trajectory viewer](https://www.swe-marathon.org/)
- [SWE-Marathon `slack-clone` task](https://github.com/abundant-ai/swe-marathon/tree/main/tasks/slack-clone)
- [SWE-Marathon `biofabric-rust-rewrite` task](https://github.com/abundant-ai/swe-marathon/tree/main/tasks/biofabric-rust-rewrite)
- [SWE-Marathon `parameter-golf` task](https://github.com/abundant-ai/swe-marathon/tree/main/tasks/parameter-golf)
- [FrontierSWE website](https://www.frontierswe.com/)
- [FrontierSWE official blog](https://www.frontierswe.com/blog)
- [FrontierSWE GitHub](https://github.com/Proximal-Labs/frontier-swe)
- [FrontierSWE scoring notes](https://github.com/Proximal-Labs/frontier-swe/blob/main/SCORING.md)
- [SWE-Together paper: arXiv 2606.29957](https://arxiv.org/abs/2606.29957)
- [SWE-Together GitHub](https://github.com/Togetherbench/SWE-Together)
- [SWE-Together Hugging Face dataset](https://huggingface.co/datasets/yfwu/SWE-Together)
- [Harbor trajectory format RFC](https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md)

