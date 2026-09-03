# 397B Agent RL 到底怎么跑：从 MCP trajectory、TITO token trace 到 Fully Async GRPO


Mercor 的 headline 是：用 1,928 个专业知识工作任务做 RL 后，Qwen3.5-397B-A17B 在 APEX-Agents 上的 Pass@1 从 16.11% 提高到 27.29%。但如果只记住“397B、GRPO、提升 69%”，几乎学不到怎么实现。

真正值得拆的是下面这条链：

```text
Harbor task directory
  → Trial.create()
  → ECR world / Modal sandbox / MCP gateway
  → ArchipelagoAgent: LLM → tool call → observation → LLM
  → TITO: token_ids + loss_mask + rollout_logprobs
  → tests/test.sh → verifier_result.rewards["reward"]
  → TITOHarborGenerator
  → 16 rollouts / prompt 的 GRPO group
  → group-relative advantage
  → DPPO token mask + prompt_mean reduction
  → Megatron 更新参数
  → NCCL 同步到 12 个 vLLM engines
```

本文不再复述“Agent RL 很难”这种背景，而是沿这条链逐段打开数据结构和控制流。读完后，至少应该能回答三个具体问题：**某个 tool observation 为什么不能有梯度？一条旧 policy 生成的 trajectory 怎么进入当前 policy 的 loss？`batch=16, samples=16, staleness=3, workers=64` 合起来到底是多少条 rollout？**

代码核查截止 **2026-09-03**，固定到 ApexAgents-SkyRL-Recipe `8e7702f`、Archipelago `da7cfef` 与本文本地核查的 SkyRL checkout。训练任务、world image 和 rubric 没有公开，因此机制与配置可以核查，headline 训练结果还不能独立复现。

<!--more-->

## 1. 一条训练样本不是 prompt，而是一个完整 task directory

SkyRL 接收的训练数据不是普通的 `prompt/reward` JSONL。公开 recipe 说明，Hugging Face dataset 的每一行只有两个核心字段：

| 字段 | 内容 |
| --- | --- |
| `path` | task 目录名，例如 `mercor-409-mk-01-c87181e6` |
| `task_binary` | 该目录的 gzip tar archive |

解压后，一个训练任务大致是：

```text
mercor-409-mk-01-c87181e6/
├── instruction.md
├── task.toml
├── archipelago.json
├── environment/
│   └── Dockerfile
└── tests/
    ├── test.sh
    ├── grade.py
    ├── verifiers.json
    ├── golden_responses.json
    └── runner/
```

这些文件不是摆设，而是一次 rollout 的五个输入面：

- `instruction.md` 是 Agent 唯一直接看到的任务文本；
- `task.toml` 控制 Agent/verifier timeout、资源和 verifier 环境变量；
- `archipelago.json` 指向 world 的 ECR image，并声明 task/world id、所需 secret、是否需要 snapshot；
- `environment/Dockerfile` 只是 Harbor 目录结构要求的占位文件，实际 world 来自预构建 ECR image；
- `tests/` 不挂载给 Agent，trial 结束后才由 Harbor 在 sandbox 内执行。

`archipelago.json` 的公开 schema 已经把关键依赖说得很直白：

```json
{
  "task_id": "task_…",
  "task_slug": "409-mk-01-c87181e6",
  "world_id": "world_…",
  "world_short_name": "management-consulting-world-409",
  "ecr_image": "<account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>",
  "required_env_keys": ["FMP_API_KEY"],
  "needs_snapshot": false
}
```

所以“一个训练样本”实际包含 prompt、初始世界、工具服务、隐藏 rubric、golden response 和 grader。只公开 prompt 远远不够复现实验；缺少 ECR image 或 `tests/`，reward function 就已经变了。

## 2. 从 task directory 到 Harbor Trial，实际发生了什么

`TITOHarborGenerator` 实现 SkyRL 的 `GeneratorInterface`。它为每条 trajectory 构造 `TrialConfig`，再调用：

```python
trial = await Trial.create(trial_config)
results = await trial.run()
```

一次 trial 的生命周期可以写成：

```text
读取 task
  → 用 archipelago.json 的 ecr_image 启动 Modal sandbox
  → 启动 world 中的 MCP servers
  → 等待 POST /mcp/ tools/list 可用
  → 必要时捕获 initial snapshot
  → 调 ArchipelagoAgent.run(instruction, context)
  → 执行 tests/test.sh / grade.py
  → 返回 Agent metadata + verifier_result
  → teardown sandbox
```

公开 `archipelago_tito.yaml` 还给出了运行时边界：Agent 最多 100 steps，tool call timeout 默认 600 秒，sandbox timeout 6000 秒；397B 启动脚本把单次 LLM timeout 设为 1800 秒、整个 Agent timeout 设为 3600 秒、单轮最多生成 40,000 token。

这解释了为什么 trajectory 耗时高度不均匀：一个样本可能两轮就结束，另一个样本可能读多个 PDF、执行 Python、修改表格，再跑到一小时 timeout。Fully async 不是锦上添花，而是在处理这种长尾。

## 3. Agent 怎么拿到 MCP tools，又怎么把 observation 送回模型

recipe 中的 `ArchipelagoAgent.run()` 不是泛泛的 ReAct 伪代码。它先连接 MCP gateway：

```python
mcp_client, connected_client, tools = await self._setup_mcp_client_and_tools()
session = connected_client.session
```

`load_mcp_tools(..., format="openai")` 把 MCP schema 转成 OpenAI tool schema。代码还会将某些只有一个 `input` 或 `request` envelope 的工具拍平，减少 tool schema 占用；真正 dispatch 时再把参数包回去。

随后构造初始消息：

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": instruction},
]
```

每一轮的控制流是：

```text
当前 TITO token stream
  → vLLM 生成 raw token IDs、text、逐 token logprob
  → vLLM parser 从文本解析 tool_calls
  → 对每个 tool call 执行 call_openai_tool(session, tc)
  → MCP content blocks 转成 role=tool messages
  → generation + observation 追加进 TITO
  → 下一轮
```

如果模型不再发 tool call，则当前 assistant message 是 final step，trajectory 结束。如果文本里出现 `<tool_call>` 却无法解析，代码不会直接当 final answer，而会注入一条纠错 observation，让模型按合法格式重试。MCP session 发生 fatal error 时，还会尝试重连一次。

一个简化的消息级 trace 是：

```json
[
  {"role": "user", "content": "读取财务模型，解释收入预测的驱动因素"},
  {"role": "assistant", "tool_calls": [{
    "id": "call_0",
    "function": {
      "name": "filesystem_search_files",
      "arguments": "{\"query\":\"*.xlsx\"}"
    }
  }]},
  {"role": "tool", "tool_call_id": "call_0", "content": "[\"/filesystem/model.xlsx\"]"},
  {"role": "assistant", "tool_calls": [{
    "id": "call_1",
    "function": {
      "name": "code_execution_code_exec",
      "arguments": "{\"code\":\"python inspect_formulas.py\"}"
    }
  }]},
  {"role": "tool", "tool_call_id": "call_1", "content": "K51 = K48*K49; K52 = ..."},
  {"role": "assistant", "content": "收入由销量、价格与区域 mix 三项驱动……"}
]
```

这只是帮助理解的数据形状，不是公开训练集中的真实样本。重要的是：Agent 的 action 不只是最后一句回答，而是所有 assistant-generated token；tool result 是环境 observation。

## 4. TITO 到底保存什么：三个等长数组，而不是一段 transcript

多轮 Agent 训练最危险的实现捷径，是先保存文本，训练时再套 chat template 重新 tokenize。工具消息、stop token、换行或 parser 版本只要有一点差异，训练端的 token 就不再是采样端实际执行的 action。

TITO（Token-In, Token-Out）直接维护四个字段：

```python
self.tokens       # prompt + 每轮生成 + 每轮 observation
self.loss_mask    # 模型生成=1，prompt/observation=0
self.logprobs     # 生成 token 的 behavior logprob，其余为 0
self.transitions  # 每一步的输入、输出、observation 与调试字段
```

初始化时：

```text
tokens    = [p0, p1, p2, p3]
loss_mask = [ 0,  0,  0,  0]
logprobs  = [ 0,  0,  0,  0]
```

假设第一轮模型实际采样出三个 token `[a0, a1, stop]`，logprob 分别是 `[-0.2, -0.7, -0.1]`；MCP observation 经 chat template 得到两个 token `[o0, o1]`。`record_step()` 后是：

```text
tokens    = [p0, p1, p2, p3, a0,   a1,   stop, o0, o1]
loss_mask = [ 0,  0,  0,  0,  1,    1,      1,  0,  0]
logprobs  = [ 0,  0,  0,  0, -0.2, -0.7, -0.1,  0,  0]
```

第二轮 final answer 生成 `[b0, b1]`：

```text
tokens    = [..., o0, o1, b0,   b1]
loss_mask = [...,  0,  0,  1,    1]
logprobs  = [...,  0,  0, -0.3, -0.4]
```

因此 policy loss 只覆盖 `a0/a1/stop/b0/b1`，不会训练模型去“生成”环境返回的 `o0/o1`。同时，下一轮 forward 仍能看到 observation，因为它保留在 `tokens` 里。

### 4.1 为什么 observation tokenization 不能简单做字符串差分

Qwen ChatML 中，模型可能已经生成了 assistant turn 的 `<|im_end|>`，而 chat template 在 tool message 前还会补换行。GLM 的 observation section 又可能以另一个 stop token 开始。直接把“完整新对话 tokenize 结果减去旧 token 数”可能导致：

- `<|im_end|>` 重复一次；
- `<|im_end|>` 后面的换行丢失；
- 历史消息被重新 tokenize，复杂度随轮数增长；
- rollout logprob 与训练 token 错一位。

TITO 的做法是预先 tokenize 一个固定 dummy base：system → user → assistant tool call。每次只渲染 `fixed_base + 当前 tool messages`，取相对 fixed base 的 delta；如果 delta 首 token 与本轮生成末尾的 stop token 重叠，就删除重复项。这样 observation tokenization 是每轮常数历史长度，而不是反复 tokenize 整段会话。

`check_invariants()` 最终强制：

```python
len(tokens) == len(loss_mask) == len(logprobs)
```

调试模式还会输出 `tito_transitions.json` 和带 special token 的 `tito_debug.txt`。这两个文件比只看自然语言 transcript 有用得多：前者定位是哪一步错位，后者能直接看到双 stop token 或缺换行。

## 5. Harbor 输出怎样变成 SkyRL trajectory

Trial 完成后，generator 从两处取数据：

```python
reward = results.verifier_result.rewards["reward"]

metadata = results.agent_result.metadata
tito_tokens = metadata["tito_tokens"]
tito_loss_mask = metadata["tito_loss_mask"]
tito_logprobs = metadata["tito_logprobs"]
```

然后找 `loss_mask` 中第一个 1，把之前的部分当 `prompt_ids`，之后全部当 `response_ids`：

```python
first_gen_idx = next(i for i, m in enumerate(tito_loss_mask) if m == 1)
prompt_ids = tito_tokens[:first_gen_idx]
response_ids = tito_tokens[first_gen_idx:]
loss_mask = tito_loss_mask[first_gen_idx:]
rollout_logprobs = tito_logprobs[first_gen_idx:]
```

注意 `response_ids` 中仍包含后续 tool observations，只是它们对应的 mask 为 0。这保证 packed forward 的上下文连续，又不会让 observation 贡献 policy gradient。

generator 最终向 SkyRL 返回对齐的 batch 字段：

```python
{
  "response_ids": [...],
  "rewards": [...],
  "loss_masks": [...],
  "rollout_logprobs": [...],
  "trajectory_generation_times": [...],
  "trajectory_time_splits": {"llm": [...], "env": [...]}
}
```

异常处理也会改变训练数据：trial 普通失败返回 dummy token、reward 0、mask 0；上下文溢出可按配置清空整条 loss mask；单轮撞到 40K `max_tokens` 且没有 tool call，则 reward 被强制为 0。换句话说，timeout、截断和 parser 错误都不是纯日志事件，它们会改变有效梯度样本分布。

## 6. reward 怎样进入 GRPO，再进入 DPPO policy loss

397B 配置中，一个 prompt 采样 16 条 trajectory。设同一 prompt 的 verifier reward 为：

```text
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

组均值是 $\bar r=0.125$。脚本设置 `grpo_norm_by_std=false`，所以不是再除以组标准差，而是直接中心化：

```text
成功 trajectory 的 advantage = 1 - 0.125 =  0.875
失败 trajectory 的 advantage = 0 - 0.125 = -0.125
```

SkyRL 将每条 trajectory 的标量 advantage 广播到其所有 `loss_mask=1` 的 token。observation token 因 mask 为 0，哪怕张量里有 advantage，也不会进入最终 loss。

### 6.1 DPPO 实际判断的是单个 token 的概率移动量

DPPO 在这里使用 `binary_tv`。对 token $t$，behavior policy 是 rollout 时的 $\mu_t$，当前训练 policy 是 $\pi_t$：

$$
r_t=\exp(\log \pi_t-\log \mu_t)=\frac{\pi_t}{\mu_t}
$$

但 DPPO 的 mask 不看 ratio 是否超过 PPO clip，而是看概率差：

$$
\Delta p_t=\pi_t-\mu_t
$$

397B 脚本设置 `delta_low=delta_high=0.15`。假设某个正 advantage token 的 rollout 概率是 $\mu_t=0.20$：

| 当前概率 | $\Delta p_t$ | 正 advantage 时 mask | 含义 |
| ---: | ---: | ---: | --- |
| $\pi_t=0.32$ | $0.12$ | 1 | 继续提高该 token 概率 |
| $\pi_t=0.38$ | $0.18$ | 0 | 已经提高太多，不再给梯度 |

若 advantage 为负，方向反过来：只有当当前概率相对 behavior policy 降低超过 0.15，才屏蔽继续降低它的梯度。

对应源码的核心就是：

```python
mask[(advantages > 0) & (prob_diff > delta_high)] = 0
mask[(advantages < 0) & (-prob_diff > delta_low)] = 0
loss = -(ratio * advantages * mask)
```

这里必须使用逐 token `rollout_logprobs`。Fully async 下，一条长 trajectory 甚至可能跨越多次权重同步才结束；如果只拿 trainer 开始更新时重算的 `old_log_probs`，就丢失了真正生成该 token 的 behavior policy。

## 7. `prompt_mean` 解决的不是“长回答”，而是 prompt group 的梯度份额

一个 training step 有 16 个 prompt groups，每组 16 条 rollout。不同知识工作任务产生的 assistant token 数可能差一个数量级。

假设：

- Prompt A 的 16 条 rollout 共 10,000 个有效 assistant token；
- Prompt B 的 16 条 rollout 共 100,000 个有效 assistant token。

`token_mean` 对全 batch token 求平均，因此 B 的总梯度份额约是 A 的 10 倍。它等价于让“更长的任务”自动获得更大的任务采样权重。

`prompt_mean` 先在每个 prompt group 内做 token mean，再对 prompt 求平均：

$$
\mathcal L=\frac{1}{2}\left(
\frac{1}{10000}\sum_{t\in A}\ell_t+
\frac{1}{100000}\sum_{t\in B}\ell_t
\right)
$$

所以 A、B 的总权重各是 $1/2$。长 trajectory 中的单个 token 权重更小，但两个任务本身仍等权。

SkyRL 的实现并不是训练后再改 scalar loss，而是在 minibatch 中预缩放 advantage。对 prompt $p$ 的 token：

$$
A'_{p,t}=\frac{A_{p,t}}{N_{prompt}\cdot N_{token,p}}
$$

随后 policy loss 仍然直接求和。这一点很关键：`prompt_mean` 需要保留 `prompt_boundaries`，不能把 256 条 trajectory shuffle 后忘记它们属于哪个 prompt。

消融中，baseline Mean Reward 为 28.69，单独改 `prompt_mean` 后是 32.54，是公开表里最清晰的单项收益。它说明训练稳定性可能首先取决于“任务怎样加权”，而不是换一个更复杂的 advantage estimator。

## 8. Fully async 一次 step 怎样运行

397B 脚本的四个数字要一起读：

```text
train_batch_size = 16 prompt groups
n_samples_per_prompt = 16 trajectories
max_staleness_steps = 3
num_parallel_generation_workers = 64 groups
```

因此一次 optimizer step 消耗：

$$
16\ \text{groups}\times16\ \text{trajectories/group}=256\ \text{trajectories}
$$

staleness manager 的 completed/running group 总 headroom 按下面的量级约束：

$$
B(S+1)=16\times(3+1)=64\ \text{groups}
$$

即最多约 $64\times16=1024$ 条 trajectory 处于这条异步管线的 running/available 容量尺度。这里的 worker 单位是 prompt group，不是单条 trajectory；每个 group 内部再并发生成 16 个 samples。

trainer 主循环按下面的顺序工作：

```text
64 个 generation group workers 持续生产
  → completed groups 放进 asyncio.Queue(maxsize=64)
  → trainer 收集 16 个有效 groups
  → 丢弃 zero-variance groups，必要时继续补齐
  → 合成 256 trajectories 的 training input
  → Megatron forward/backward/optimizer step
  → pause generation
  → NCCL broadcast 新权重到 vLLM engines
  → increment weight_version
  → resume generation
  → staleness manager 放出新的 submission slots
```

`sample_full_batch=true` 与 `zero_variance_filter=true` 配套：如果一个 prompt 的 16 个 reward 没有方差，GRPO 中心化后整组 advantage 都接近 0，这组数据没有有效学习信号；trainer 丢掉它并继续从 buffer 取组，直到凑齐 16 个有效 groups。

### 8.1 staleness=3 不是逐条 trajectory 的硬 SLA

staleness manager 使用的是全局容量规则：

```python
consumer_capacity = (max_staleness_steps + current_global_step) * mini_batch_size
capacity = consumer_capacity - (accepted + running)
```

这在稳态限制生成端不要领先 trainer 太多，但不能保证每条长尾 trajectory 都在 3 个 step 内结束。某个任务如果跑了很久，完成时仍可能超过 staleness budget；SkyRL 会接受它、记录 violation 并报警，而不是必然丢弃。

### 8.2 权重同步时正在生成的请求怎么办

公开代码的共同原则是：不能让一次请求读到“同步一半”的权重，所以 `save_weights_for_sampler()` 把同步包在 pause/resume 之间。具体怎样处理 in-flight KV 取决于 backend 与版本：

- 新版本 `offload_kv_for_weight_sync` 可暂停请求、把 KV offload 到 CPU、同步权重后恢复，避免重新 prefill；
- 本次公开 397B 脚本固定的 OSS SkyRL 版本没有这个 knob，因此脚本明确省略；
- 若没有 KV offload，代码仍 pause → broadcast → resume，但显存余量和 in-flight 行为不能直接等同 production hero run。

脚本把 vLLM `gpu_memory_utilization` 设为 0.93，同时提示没有 KV offload 时若 weight sync OOM，应降到约 0.84。这是一个实质性的公开/生产差异，不是注释层面的无关细节。

## 9. 20×8 H200 的 397B 脚本逐项拆开

总资源是 20 个节点、每节点 8 张 H200，共 160 GPU。训练和生成不 colocate：

| 区域 | GPU | 配置 | 实际职责 |
| --- | ---: | --- | --- |
| vLLM rollout | 12×8 = 96 | 12 engines，每个 TP8 | 运行 64 个 group workers 发来的采样请求 |
| Megatron trainer | 8×8 = 64 | TP4 / PP4 / CP2 / EP16 / ETP1 | 397B MoE forward、backward、optimizer |

Megatron 的 64-way model parallel 不是把这些维度简单相乘。TP4、PP4、CP2 形成 dense/sequence 路径的切分，EP16 切 MoE experts；这些并行维度存在嵌套和正交关系。脚本注释给出的结果是 64-way model parallel、DP=1，即这 64 张训练卡没有额外 data-parallel replica。

关键参数及其工程含义：

| 参数 | 值 | 为什么这样设 |
| --- | ---: | --- |
| train context | 160,000 | 决定训练 forward 的最大序列与 microbatch token budget |
| eval/vLLM max len | 262,144 | 给评测更长上下文；训练仍截在 160K |
| per-turn max tokens | 40,000 | 防止单次 LLM call 独占整段上下文 |
| max concurrency | 300 trajectories | 保护 Modal/world/API 与推理端不被瞬时并发压垮 |
| rate limit | 3 trajectories/s | 控制新 trial 启动速率，不等于每秒完成 3 条 |
| temperature | 1.0 | 为同 prompt 的 16 samples 提供探索差异 |
| learning rate | $10^{-6}$ | 超大模型 RL 的保守更新尺度 |
| Adam betas | $[0.9,0.98]$ | 第二动量比常见 0.999 更快响应近期梯度变化 |
| weight decay | 0.01 | 常规参数正则 |
| max grad norm | 1.0 | 限制极端 batch 的梯度范数 |
| KL loss | false | 不另加 reference-model KL penalty |
| DPPO delta | 0.15 / 0.15 | 用 token probability movement 做双向 mask |
| `prompt_mean` | true | 每个 prompt group 等权 |
| `grpo_norm_by_std` | false | advantage 只减组均值，不除组标准差 |
| optimizer offload | 100% | 把 optimizer state 压到 CPU，换取 GPU 容量 |
| checkpoint | S3 | 397B 多节点 checkpoint 不能依赖单节点本地盘重组 |

Qwen3.5-397B-A17B 是 MoE，单 token 激活约 17B 参数；但 checkpoint、optimizer state、专家分片和 trainer→12 个 engines 的同步仍面对 397B 总参数规模。A17B 不能把这次系统问题降格成普通 17B 训练。

## 10. 最小能验证到哪一步，以及会在哪里卡住

公开 recipe 给出的最小闭环是 1 GPU smoke：

```bash
git clone https://github.com/NovaSky-AI/SkyRL /mnt/local_storage/oss/SkyRL-v0.3.0
git -C /mnt/local_storage/oss/SkyRL-v0.3.0 checkout b8a5caaa
uv sync
bash scripts/run_1gpu_colocated_smoke.sh
```

默认用 Qwen3.5-0.8B、$2\times2$ batch、跑 2 steps；它想验证的是：

```text
Trial → TITO rollout → verifier reward → DPPO backward → checkpoint
```

但 fresh user 会在 data 层先卡住。公开仓库没有 `apex-agents-dev-1928` 的 task archives，也没有对应 112 个 ECR worlds、任务 secret、隐藏 `verifiers.json` 与 `golden_responses.json`。所以严谨的复现分三层：

1. **源码机制可核查**：TITO 数组、GRPO、DPPO、`prompt_mean`、buffer 与 staleness 都能读代码和写单测；
2. **自建任务可跑通**：按相同 Harbor 目录 schema 自己做一个 toy task/world/verifier，可验证端到端接口；
3. **Mercor headline 不可独立重跑**：缺私有训练 task、world images、grader 资产与完整集群记录。

实际调试时，我会按这个顺序验收：

```text
先跑单条 Harbor task
  → 检查 trajectory.json 的 tool call / observation
  → 检查 tito_transitions.json
  → 断言三数组等长，observation mask=0
  → 手算一组 reward 的 GRPO advantage
  → 单 batch 过拟合 32 个有 reward variance 的 prompts
  → 最后才开 fully async 和多节点 weight sync
```

直接启动 397B 脚本无法替代这些检查。大集群只会把 token 错位、grader 漂移或全零 advantage 更快地扩散。

## 11. 结果：哪些数字值得信，哪些不能替实现细节

主结果如下：

| 模型 | APEX Pass@1 | 绝对提升 |
| --- | ---: | ---: |
| Qwen3.6-35B-A3B | 13.96% → 22.71% | +8.75 pp |
| Qwen3.5-397B-A17B | 16.11% → 27.29% | +11.18 pp |

跨任务的 Terminal-Bench 2.1 也上涨：35B 从 44.57 到 50.94，397B 从 50.56 到 55.43。公开的 1,068 条 TBench JSON trace 可以独立复算这些 Pass@1；APEX 主 trace 约 22.6 GB，但 gated，匿名条件下不能完成相同复算。

35B 消融最有用的几行是：

| 配方 | Mean Reward | Pass@1 |
| --- | ---: | ---: |
| baseline | 28.69 | 13.12 |
| DPPO | 29.03 | 13.96 |
| `prompt_mean` | **32.54** | **16.74** |
| context nudge | 31.64 | 15.69 |
| DPPO + `prompt_mean` + nudge | 31.81 | 16.11 |

这张表不支持“DPPO 是主要涨点来源”：单项只增加 0.34 Mean Reward。它更支持两件事：prompt-level loss weighting 很重要；组件组合存在交互，单项增益不能相加。

## 12. Archipelago 到底负责什么：一句话够了

[Archipelago](https://github.com/Mercor-Intelligence/archipelago/tree/da7cfef40f3f5cd2fb298055d43fa14ef0a8f7d9) 主仓提供 environment、Agent runner、grading runner 和端到端 examples。它定义了“可操作 world + MCP tools + snapshot/artifact grading”的任务形态。

本次训练真正使用的是：Harbor 管 trial 生命周期；recipe 自己的 `ArchipelagoAgent` 复现 MCP loop 并加入 TITO；每个私有 task 的 `tests/runner/` vendored 了一份 Archipelago grading code；SkyRL 管生成队列、GRPO/DPPO、Megatron 与权重同步。

所以分析边界只需记住：**Archipelago 定义工作与评分界面，recipe 把这个界面接进 RL，SkyRL 才是训练系统。**

## 最后带走一条具体判断

这份公开工作的最高价值，不是证明“397B 做 Agent RL 会涨点”，而是给出了长程 tool-use RL 中一条完整、可审计的数据契约：

```text
模型实际采样的 token
  + 当时的逐 token logprob
  + 明确区分 action/observation 的 loss mask
  + task verifier 给出的 outcome reward
  + prompt group 身份
  + 生成时的 policy version
```

少其中任何一项，GRPO/DPPO 的公式即使写对，也可能没有在训练“Agent 实际做过的事”。

## 参考资料

- [Mercor：Training frontier knowledge work agents: A 397B RL training guide with SkyRL](https://www.mercor.com/blog/training-frontier-knowledge-work-agents-a-397b-rl-training-guide-with-skyrl/)，核查日期：2026-09-03。
- [Mercor-Intelligence/ApexAgents-SkyRL-Recipe](https://github.com/Mercor-Intelligence/ApexAgents-SkyRL-Recipe/tree/8e7702f03b7464a36ab800a624fd911de0968a87)，核查提交：`8e7702f`。
- [Recipe：TITO state](https://github.com/Mercor-Intelligence/ApexAgents-SkyRL-Recipe/blob/8e7702f03b7464a36ab800a624fd911de0968a87/apex_agents_skyrl_recipe/agents/tito.py)。
- [Recipe：ArchipelagoAgent MCP loop](https://github.com/Mercor-Intelligence/ApexAgents-SkyRL-Recipe/blob/8e7702f03b7464a36ab800a624fd911de0968a87/apex_agents_skyrl_recipe/agents/archipelago.py)。
- [Recipe：TITOHarborGenerator](https://github.com/Mercor-Intelligence/ApexAgents-SkyRL-Recipe/blob/8e7702f03b7464a36ab800a624fd911de0968a87/apex_agents_skyrl_recipe/tito_harbor_generator.py)。
- [Recipe：397B fully async 启动脚本](https://github.com/Mercor-Intelligence/ApexAgents-SkyRL-Recipe/blob/8e7702f03b7464a36ab800a624fd911de0968a87/scripts/run_qwen35_397b_fully_async.sh)。
- [Recipe：完整消融表](https://github.com/Mercor-Intelligence/ApexAgents-SkyRL-Recipe/blob/8e7702f03b7464a36ab800a624fd911de0968a87/assets/full_ablation_table.md)。
- [Mercor-Intelligence/archipelago](https://github.com/Mercor-Intelligence/archipelago/tree/da7cfef40f3f5cd2fb298055d43fa14ef0a8f7d9)，核查提交：`da7cfef`。
- [SkyRL：Fully Async Training](https://docs.skyrl.ai/docs/tutorials/fully_async) 与 [Off-policy Correction](https://docs.skyrl.ai/docs/algorithms/off_policy_correction)。
- [APEX-Agents paper](https://arxiv.org/abs/2601.14242) 与 [DPPO paper](https://arxiv.org/abs/2602.04879)。
- [Terminal-Bench 2.1 eval traces](https://huggingface.co/datasets/mercor/ApexAgentsRecipe-TBench2_1-EvalTraces)。

