# GRPO 训练 DeepSeek-V3 的数学能力：从 rollout 到参数更新


假设模型面对一道数学题，四次生成中只有一次答对。训练系统不会把标准答案当作交叉熵标签直接喂回模型，而是把四份回答放在一起比较：答对的回答得到正的相对优势，答错的回答得到负的相对优势，再通过当前模型对这些 token 的概率计算梯度。

这就是本文要复原的核心链路：

```text
prompt
  → 旧策略采样多份回答
  → reward 函数打分
  → 组内标准化得到 advantage
  → 当前策略重新计算 log-probability
  → ratio + clip + KL 构成 loss
  → backward() 更新 actor 参数
```

本文只讨论**可验证最终答案的数学任务**，并使用教学设定解释机制。DeepSeek-V3 的完整训练还包含预训练、SFT、代码与写作等任务，以及从 DeepSeek-R1 系列蒸馏推理能力的后训练流程；下面的数学例子不是官方训练配置的逐项复现。[DeepSeek-V3 技术报告，第 5 节](https://arxiv.org/html/2412.19437v2#S5)

<!--more-->

## 1. 30 秒结论

先记住四个判断。

1. **GRPO 调整的是生成倾向，不是把答案写入参数。** 模型先自己生成回答，reward 只负责给回答评分，真正参与反向传播的是当前策略对已生成 token 的 log-probability。
2. **advantage 是组内相对量。** 同一道题生成 $G$ 份回答，用这一组回答的分数估计 baseline；一组全对或全错时，reward 驱动的 advantage 会变成零。
3. **ratio 让旧样本可以用于当前策略的局部更新。** 它比较当前模型与采样时旧模型对同一个 token 的概率，clip 限制在旧样本上过度更新，KL 则约束当前策略不要偏离参考模型太远。
4. **这条链路不等于 DeepSeek-V3 未公开代码的精确复现。** V3 报告和 DeepSeekMath 论文在目标函数写法、回答级与 token 级 ratio 等细节上存在需要区分的表述；本文采用 DeepSeekMath 中更适合逐 token 解释的教学形式。

DeepSeek-V3 是总参数约 671B、每个 token 激活约 37B 参数的 MoE 模型；官方材料说明其训练路线包括预训练、SFT 和 RL。[官方模型介绍](https://github.com/deepseek-ai/DeepSeek-V3) 这些架构数字说明了 actor 的规模，但不能据此推断某一次数学 RL 更新会同时激活固定的 37B 参数：MoE 路由会随 token 和专家选择变化。

## 2. 读完后应能回答

- 一条数学 RL 数据中，题目、标准答案、模型回答分别进入哪个组件？
- 为什么 GRPO 不需要单独训练 critic/value model？
- reward 如何变成组内 advantage，又如何作用到每个回答 token？
- 第一次更新时 ratio 数值可以是 $1$，为什么仍然有梯度？
- 旧策略、当前策略和参考策略什么时候固定，什么时候刷新？
- 哪些结论可以由这个算例支持，哪些不能外推成真实 V3 配置？

## 3. 概念与符号

| 符号 / 概念 | 含义 | 本文中的边界 |
| --- | --- | --- |
| $q$ | 数学题 prompt | 输入不包含标准答案 |
| $o_i$ | 第 $i$ 份完整回答 | 由旧策略采样得到 |
| $G$ | 每道题的回答数量 | 不是更新次数 |
| $\pi_\theta$ | 当前正在训练的 actor | 参数会被更新 |
| $\pi_{\mathrm{old}}$ | 采样当前 rollout 的旧策略 | 当前 rollout 内固定 |
| $\pi_{\mathrm{ref}}$ | KL 约束使用的参考策略 | 通常固定或慢速更新 |
| $r_i$ | 第 $i$ 份回答的 reward | 本例只检查最终答案和格式 |
| $A_i$ | 组内标准化后的 advantage | 最终结果监督下广播到回答 token |
| $\rho_{i,t}$ | 当前策略与旧策略对 token $t$ 的概率比 | 是局部 token ratio，不是整段完整比值 |
| rollout | 从题目开始到回答结束的一次采样轨迹 | 状态是前缀，动作是下一个 token |

最简的语言模型 rollout 可以写成：

$$
s_t=(q,o_{<t}),\qquad
a_t=o_t,\qquad
s_{t+1}=(q,o_{\leq t}).
$$

也就是说，状态是题目加已经生成的前缀，动作是下一个 token，生成结束或达到长度上限后得到完整回答，再由 reward 函数评分。

## 4. 真正的问题：如何让高分回答更容易出现

### 4.1 这不是从零开始靠 reward 学数学

DeepSeek-V3 的路线是预训练、SFT 和 RL。RL 阶段面对的是一个已经具备语言和一定解题能力的模型；它要改变的是不同解法、格式和行为的概率分布，而不是从空白网络中直接写出数学知识。

设 SFT 后的模型参数为 $\theta_0$。对同一道题，它可能生成多种回答：

- 推导和最终答案都正确；
- 中间计算错误；
- 推导基本正确，但最终格式不符合判分器要求；
- 过程看起来合理，最终答案却错误。

GRPO 的目标是提高高 reward 回答的相对概率，降低低 reward 回答的相对概率。它不等于把 `reference_answer` 拼接到 prompt 中，也不等于用 reward 替代语言模型的交叉熵标签。

### 4.2 一条数据中各字段去哪里

可以把一条训练数据写成：

```json
{
  "id": "math_001",
  "prompt": "3盒苹果，每盒12个，吃掉5个，还剩几个？请将最终答案写在\\boxed{}中。",
  "reference_answer": "31"
}
```

两个字段进入不同组件：

```text
prompt ───────────────→ actor，供模型生成回答
reference_answer ─────→ reward 函数，用来检查最终答案
```

| 训练方式 | 训练时的解答从哪里来 |
| --- | --- |
| SFT | 数据集预先提供目标解答，模型学习模仿 |
| 这里的 GRPO | 模型自己生成回答，判分器返回分数 |

因此，数学 RL 数据可以只提供题目和可验证的标准答案，而不提供标准解题过程。但不能由此说 V3 的训练完全不需要推理数据：它此前的 SFT，以及 V3 报告中提到的推理能力蒸馏，都属于其他训练阶段。

## 5. 第一步：旧策略 rollout

### 5.1 三个策略角色

为了避免把不同模型版本混在一起，先区分三个逻辑角色：

| 角色 | 作用 | 当前这批 rollout 内是否更新 |
| --- | --- | --- |
| 当前策略 $\pi_\theta$ | 正在训练的 actor | 是 |
| 旧策略 $\pi_{\mathrm{old}}$ | 生成当前回答的策略版本 | 否 |
| 参考策略 $\pi_{\mathrm{ref}}$ | 提供 KL 约束的参考版本 | 通常否 |

“三个角色”不等于显存中必须常驻三份完整 V3。旧策略可以由生成副本承担，也可以在采样后只保留已生成 token 的 `old_logp`；工程实现取决于推理服务、并行方式和显存预算。

### 5.2 同一道题生成四份回答

设这一轮开始时：

$$
\theta=\theta_0,\qquad
\theta_{\mathrm{old}}=\theta_0.
$$

用同一个策略对苹果题随机采样 $G=4$ 次：

| 回答 | 生成内容 | 最终答案 |
| --- | --- | --- |
| $o_1$ | `3×12=36，36-5=31` | `\boxed{31}` |
| $o_2$ | `3×12=36，36+5=41` | `\boxed{41}` |
| $o_3$ | `12+3=15，15-5=10` | `\boxed{10}` |
| $o_4$ | `3×12=35，35-5=30` | `\boxed{30}` |

这里的 $G$ 是每道题的回答数量，不是优化器更新次数。若一次采样 $B$ 道题，总回答数是 $BG$。

生成阶段通常保存：

```text
题目 token
回答 token
回答有效位置 mask
生成时的 old_logp
```

对回答中的第 $t$ 个 token：

$$
\ell^{\mathrm{old}}_{i,t}
=
\log\pi_{\theta_{\mathrm{old}}}
(o_{i,t}\mid q,o_{i,<t}).
$$

生成阶段通常不保留用于反向传播的计算图。后续训练会把已经采样的文本重新喂给当前 actor，重新计算带梯度的 log-probability。

## 6. 第二步：reward 与组内 advantage

### 6.1 一个最小 reward 函数

只看整数最终答案时，可以用如下教学版判分器：

```python
import re

def reward_fn(text, reference_answer):
    match = re.search(
        r"\\boxed\{([+-]?\d+)\}\s*$",
        text.strip(),
    )
    return float(
        match is not None
        and int(match.group(1)) == int(reference_answer)
    )
```

于是四份回答的 reward 是：

$$
[r_1,r_2,r_3,r_4]=[1,0,0,0].
$$

这个函数只适合当前例子，不能当成通用数学判分器。分数、小数、集合和代数表达式可能有不同但等价的写法，需要解析器、规范化或符号等价性检查。

还要注意：这个 reward 没有逐步检查推理。如果一份回答的过程错误，但最终碰巧写出 31，它仍可能得到 1。因此 outcome reward 对信用分配很粗，不能单独证明模型学会了正确的解题过程。

### 6.2 GRPO 为什么不需要 critic

GRPO 的关键做法是用同一道题生成的一组回答估计相对 baseline，而不是像 PPO 那样额外训练一个 value/critic 模型。[DeepSeekMath 论文](https://arxiv.org/html/2402.03300v3#S4.SS1) 将其描述为 Group Relative Policy Optimization，并报告了不使用 critic 的设计。

对同一道题的 $G$ 个 reward 做组内标准化：

$$
A_i
=
\frac{r_i-\bar r}{s_r+\delta},
\qquad
\bar r=\frac{1}{G}\sum_{j=1}^{G}r_j.
$$

其中 $s_r$ 是这一组 reward 的标准差，$\delta$ 是防止除零的小常数。本例采用样本标准差并忽略极小的 $\delta$：

$$
\bar r=0.25,\qquad s_r=0.5,
$$

所以：

$$
[A_1,A_2,A_3,A_4]=[1.5,-0.5,-0.5,-0.5].
$$

含义很直接：第一份回答比同组平均表现好，另外三份比平均表现差。

**比较发生在同一道题内。** 不能把完全不同题目的 reward 直接混在一起做这个组内比较。

### 6.3 同一个 advantage 如何作用到多个 token

在只提供最终结果的监督下，同一份回答的所有有效回答 token 通常共享同一个 advantage：

$$
A_{i,t}=A_i.
$$

因此，$o_1$ 中的 `3`、`×`、`12`、`=` 等 token 都收到 $+1.5$ 的权重；$o_2,o_3,o_4$ 的回答 token 都收到 $-0.5$ 的权重。

这并不意味着算法知道哪一步是决定性步骤。它只知道整条回答的最终分数，所以信用分配比过程监督更粗。

如果一组回答全对或全错，所有 reward 相同，则：

$$
A_i=0.
$$

此时这组数据的 reward 驱动策略梯度消失；如果目标里还有 KL 项，模型仍可能收到 KL 梯度。

## 7. 第三步：当前策略重新前向

现在四份回答、四个 advantage 都已经固定。训练阶段不再重新采样，而是把题目和回答喂回当前 actor，得到每个实际生成 token 的当前 log-probability：

$$
\ell_{i,t}
=
\log\pi_\theta(o_{i,t}\mid q,o_{i,<t}).
$$

因果注意力保证位置 $t$ 的输出只能使用题目和之前的前缀，不能偷看后面的 token。

### 7.1 取出实际 token 的 log-probability

模型通常返回：

```text
logits: [batch, sequence_length, vocabulary_size]
```

需要先做 `log_softmax`，再取实际 target token 对应的词表位置：

```python
def token_logps(model, input_ids, attention_mask):
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits

    log_probs = logits[:, :-1].log_softmax(dim=-1)
    targets = input_ids[:, 1:]

    return log_probs.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)
```

错开一位是因为位置 $t$ 的输出预测位置 $t+1$ 的 token。loss 只应计入回答有效位置，排除 prompt 和 padding，通常也包含回答结束标记。

“prompt 位置不直接计算 loss”不等于“prompt 的内部计算没有梯度”：回答 token 的概率依赖题目的隐藏表示，梯度仍然可以沿这条依赖回传。

## 8. 第四步：ratio、clip 与 KL

### 8.1 ratio 比较当前策略和旧策略

当前回答是旧策略采样出来的，但同一批回答可能被重复更新多次，因此需要比较当前策略是否已经偏离采样策略：

$$
\rho_{i,t}
=
\frac{
\pi_\theta(o_{i,t}\mid q,o_{i,<t})
}{
\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})
}
=
\exp\left(
\ell_{i,t}-\ell^{\mathrm{old}}_{i,t}
\right).
$$

例如旧模型在同一个前缀下生成 `−` 的概率为 $0.60$，当前模型对应概率为 $0.66$，则：

$$
\rho=\frac{0.66}{0.60}=1.1.
$$

这表示当前模型选择该 token 的概率是采样时的 $1.1$ 倍。

从重要性采样的角度，对固定前缀和函数 $f$：

$$
\begin{aligned}
\mathbb E_{a\sim\pi_\theta(\cdot\mid s)}[f(a)]
&=\sum_a\pi_\theta(a\mid s)f(a)\\
&=\mathbb E_{a\sim\pi_{\mathrm{old}}(\cdot\mid s)}
\left[
\frac{\pi_\theta(a\mid s)}
{\pi_{\mathrm{old}}(a\mid s)}
f(a)
\right].
\end{aligned}
$$

直觉是：手上的动作来自旧模型，但可以按照新模型会多常选择它重新加权。

这里必须保留一个边界：逐 token 的 $\rho_{i,t}$ 不是整条回答分布变化的完整校正。整段回答的概率比是：

$$
\rho_i^{\mathrm{seq}}
=
\frac{\pi_\theta(o_i\mid q)}
{\pi_{\mathrm{old}}(o_i\mid q)}
=
\prod_t\rho_{i,t}.
$$

逐 token clipped surrogate 与先把 ratio 连乘、再对整段 clip 并不等价。本文采用的是 DeepSeekMath 中适合解释 token 级梯度的形式，不能把它直接称为 DeepSeek-V3 未公开实现的逐行复现。

### 8.2 policy loss 与 clip

忽略 clip 和 KL 时，策略项可以写成：

$$
\ell^{\mathrm{policy}}_{i,t}=-\rho_{i,t}A_i.
$$

为了避免在同一批旧样本上沿同一方向更新过头，引入 $\epsilon$-clip：

$$
\ell^{\mathrm{policy}}_{i,t}
=
-\min\left(
\rho_{i,t}A_i,\;
\operatorname{clip}
(\rho_{i,t},1-\epsilon,1+\epsilon)A_i
\right).
$$

取教学设定 $\epsilon=0.2$，看四个独立 token 的数值：

| $A$ | 旧概率 | 当前概率 | $\rho$ | clipped policy loss |
| ---: | ---: | ---: | ---: | ---: |
| $+1.5$ | $0.60$ | $0.66$ | $1.10$ | $-1.65$ |
| $+1.5$ | $0.60$ | $0.75$ | $1.25$ | $-1.80$，进入平台 |
| $-0.5$ | $0.30$ | $0.27$ | $0.90$ | $+0.45$ |
| $-0.5$ | $0.30$ | $0.21$ | $0.70$ | $+0.40$，进入平台 |

clip 不会硬性保证 $\rho$ 永远落在 $[0.8,1.2]$ 内。它只是改变目标函数；参数共享、其他样本和 KL 梯度仍可能把实际概率推到区间之外。

### 8.3 KL 把策略拉回参考模型附近

参考模型对同一个 token 的 log-probability 为：

$$
\ell^{\mathrm{ref}}_{i,t}
=
\log\pi_{\mathrm{ref}}(o_{i,t}\mid q,o_{i,<t}).
$$

一种常见的采样 KL 惩罚写作：

$$
\Delta_{i,t}
=
\ell^{\mathrm{ref}}_{i,t}-\ell_{i,t},
\qquad
k_{i,t}
=
\exp(\Delta_{i,t})-\Delta_{i,t}-1.
$$

于是单个 token 的总损失为：

$$
\ell_{i,t}
=
\ell^{\mathrm{policy}}_{i,t}
+
\beta k_{i,t}.
$$

策略项推动模型朝高 reward 方向改变，KL 项限制这种改变不要过度偏离参考策略。

这是一种基于采样 token 的 KL 惩罚，不是每个 token 上已经精确求完词表期望的 KL；复用旧策略样本时，也不能无条件把它称为当前策略 KL 的无偏估计。

### 8.4 汇总成一个交给 backward 的标量

对一组回答，先按每个回答的有效 token 数归一化，再对回答取平均：

$$
L(\theta)
=
\frac{1}{G}
\sum_{i=1}^{G}
\frac{1}{|o_i|}
\sum_{t=1}^{|o_i|}
\left[
-\min\left(
\rho_{i,t}A_i,\;
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i
\right)
+\beta k_{i,t}
\right].
$$

如果一次有多道题，再对题目平均。到这里，才得到交给 `backward()` 的 loss。

## 9. 第五步：梯度如何改变模型参数

### 9.1 reward 不直接反向传播

先忽略 clip 和 KL，对一个 token：

$$
\ell=-A\rho.
$$

由于 $A$ 和旧策略概率固定：

$$
\nabla_\theta\ell
=
-A\rho\,
\nabla_\theta
\log\pi_\theta(o_t\mid q,o_{<t}).
$$

因此梯度路径是：

```text
GRPO loss
   ↓
ratio / KL 中的当前 log-probability
   ↓
log-softmax 与输出 logits
   ↓
输出层、Attention、MoE 等计算
   ↓
当前 actor 的可训练参数 θ
```

不会沿这条路径更新的包括：

```text
已采样的 token ID
reward 函数
advantage 数值
缓存的 old_logp
冻结参考模型参数
```

reward 通过 $A$ 决定梯度方向和权重，但不是从 reward 函数内部对离散采样过程求导。

### 9.2 一个 logit 的局部计算

设被采样 token 是 $a$，其当前 logit 为 $z_a$，概率为 $p_a$。有：

$$
\frac{\partial\log p_a}{\partial z_a}=1-p_a.
$$

所以：

$$
\frac{\partial\ell}{\partial z_a}
=
-A\rho(1-p_a).
$$

第一次更新前 $\rho=1$。对高 reward 回答中的某个 token：

$$
A=1.5,\qquad p_a=0.6,
$$

则：

$$
\frac{\partial\ell}{\partial z_a}
=
-1.5(1-0.6)=-0.6.
$$

梯度下降会倾向于提高该 token 的 logit。对低 reward 回答中的某个 token：

$$
A=-0.5,\qquad p_a=0.3,
$$

则：

$$
\frac{\partial\ell}{\partial z_a}
=
-(-0.5)(1-0.3)=0.35,
$$

梯度下降会倾向于降低该 token 的 logit。

这只是单个 loss 项的局部作用，不保证更新后每个高 reward token 的概率都上升。所有回答共享网络参数，同时还存在其他 token 的梯度和 KL 约束。

### 9.3 为什么第一次 loss 可以为零但仍有梯度

第一次更新前，当前策略和旧策略相同，因此：

$$
\rho_{i,t}=1.
$$

忽略 KL，四份回答的 policy loss 平均值是：

$$
-\frac{1}{4}(1.5-0.5-0.5-0.5)=0.
$$

但不同回答对应的是不同的 log-probability 函数，它们的导数并不会因为数值权重相加为零就自动抵消。

代码中的 ratio 也不能直接替换成常数 1：

```python
ratio = (current_logp - old_logp).exp()
```

第一次更新时它的数值可能是 1，但它仍依赖带计算图的 `current_logp`。如果写成 `ratio = 1.0`，这条梯度路径就被切断了。

## 10. 旧策略什么时候刷新

假设一批 rollout 数据执行两次完整参数更新：

| 时刻 | 使用的数据 | 当前参数 | 这批数据的旧策略 |
| --- | --- | --- | --- |
| 生成批 A | 新生成的回答 | $\theta_0$ | $\theta_0$ |
| 第一次更新前 | 批 A | $\theta_0$ | $\theta_0$ |
| 第二次更新前 | 仍是批 A | $\theta_1$ | 仍是 $\theta_0$ |
| 生成批 B | 重新采样回答 | $\theta_2$ | 改成 $\theta_2$ |

第二次更新时，回答 token、reward、advantage 和 `old_logp` 都不变；只有当前 log-probability 重新计算，因此 ratio、clip 状态和梯度会变化。

完成规定的更新后，actor 才用新参数重新 rollout，新的回答才绑定新的旧策略版本。

还要区分三个容易混淆的概念：

- rollout batch：一次采样得到的数据；
- mini-batch：把 rollout batch 切开后用于前向的子批次；
- 梯度累积：多个 mini-batch 的梯度累积后再执行一次 `optimizer.step()`。

看到一次 `forward()` 和 `backward()`，不能据此判断旧策略应该刷新。

## 11. 一个最小的教学版训练循环

下面的代码展示一个 prompt、四份回答和两次更新。它刻意省略了分布式切分、推理服务、显存优化和真实数据管线，因此不是可直接训练 671B 模型的脚本。

```python
G = 4
updates_per_rollout = 2

clip_eps = 0.2
beta = 0.04  # 教学设定，不是声称 DeepSeek-V3 官方使用这个值

# actor：从已有模型初始化，参数可训练
# reference：本阶段固定的参考模型
# optimizer：只管理 actor 的可训练参数

for example in dataset:
    prompt = example["prompt"]
    gold = example["reference_answer"]

    # 采样和 old/ref logp 固定，不保留 actor 反向图。
    with torch.no_grad():
        ids, attention_mask, response_mask, texts = sample(
            actor,
            prompt,
            num_generations=G,
            temperature=1.0,
        )

        old_logp = token_logps(actor, ids, attention_mask)
        ref_logp = token_logps(reference, ids, attention_mask)

        rewards = torch.tensor(
            [reward_fn(text, gold) for text in texts],
            device=ids.device,
        )

        advantages = (
            (rewards - rewards.mean())
            / (rewards.std(unbiased=True) + 1e-8)
        )
        advantages = advantages[:, None]  # 广播到回答 token

    for _ in range(updates_per_rollout):
        optimizer.zero_grad()

        # 重新前向，保留当前 actor 的计算图。
        logp = token_logps(actor, ids, attention_mask)

        ratio = (logp - old_logp).exp()

        direct = ratio * advantages
        clipped = ratio.clamp(
            1 - clip_eps,
            1 + clip_eps,
        ) * advantages

        delta = ref_logp - logp
        kl = delta.exp() - delta - 1

        token_loss = -torch.minimum(direct, clipped) + beta * kl

        # 排除 prompt 和 padding；先按回答平均，再按组平均。
        sequence_loss = (
            (token_loss * response_mask).sum(dim=-1)
            / response_mask.sum(dim=-1)
        )
        loss = sequence_loss.mean()

        loss.backward()
        optimizer.step()

    # actor 已更新。
    # 下一轮重新采样，并保存对应的新 old_logp。
```

这个循环能否提升数学能力，最终要看**独立数学评测**上的正确率，而不是只看训练题 reward。loss、KL、熵和回答长度有诊断价值，但都不能单独替代正确率；不同 rollout 批次之间的 loss 也不能像普通 SFT 交叉熵那样直接比较。

## 12. 实验证据：哪些结论被支持

### 问题一：去掉 critic 是否仍能做 RL？

- **对照：** PPO 风格的 critic/value baseline 与 GRPO 的组内相对分数。
- **证据：** DeepSeekMath 将 GRPO 定义为不使用 critic、而由 group scores 估计 baseline 的方法，并报告其在数学任务上的提升。[DeepSeekMath §1 与 §4.1](https://arxiv.org/html/2402.03300v3#S4.SS1)
- **支持：** 组内比较可以构成一种资源更省的 baseline 估计方式。
- **不能说明：** 不需要 critic 就意味着所有任务、所有 reward 设计都能稳定训练。组大小、reward 方差、任务难度和数据质量仍然重要。

### 问题二：V3 的后训练是否只有数学 GRPO？

- **对照：** 把数学教学例子与 V3 的完整后训练路线区分开。
- **证据：** V3 技术报告描述了预训练后的 SFT 与 RL，并讨论了推理能力蒸馏；官方仓库同时给出 671B 总参数、37B 激活参数和 14.8T 预训练 token 等模型摘要信息。
- **支持：** 本文的数学 GRPO 是完整训练流程中的一个抽象子问题。
- **不能说明：** `G=4`、`beta=0.04`、两次 update 或苹果题 reward 是 V3 官方配置。它们只是为了让梯度链路可手算。

### 问题三：reward 是否证明模型学会了正确推理？

- **对照：** outcome reward 只检查最终答案，与逐步 process supervision 区分。
- **证据：** 本文的 reward 函数即使遇到错误过程但正确最终答案，也会返回 1。
- **支持：** outcome reward 可以推动最终结果变好。
- **不能说明：** 模型一定学会了可靠、可解释且每一步都正确的解法。

## 13. Claim–evidence map

| Claim | 直接证据 | 强度 | 不能外推成什么 |
| --- | --- | --- | --- |
| GRPO 可以用组内分数估计 baseline，不依赖单独 critic | DeepSeekMath 对 GRPO 的方法描述 | 来源结论 | 所有 reward 任务都不需要 value model |
| V3 的训练包含预训练、SFT 与 RL | DeepSeek-V3 技术报告第 1、5 节 | 来源结论 | V3 数学阶段等同于本文教学循环 |
| 高 reward 回答的 token 会获得正向策略权重 | policy-gradient 目标与局部梯度推导 | 综合判断，基于目标函数 | 每个高 reward token 更新后概率都必然增加 |
| 第一次 ratio 数值为 1 仍然有梯度 | ratio 对当前 log-probability 的依赖 | 数学推导 | 把 ratio 写成常数 1 也能得到同样结果 |
| outcome reward 的信用分配较粗 | 同一回答的 advantage 广播到所有回答 token | 机制推导 | 训练出的每一步推理都一定正确 |

## 14. 局限与风险

### 14.1 教学超参数不是官方配置

本文使用 $G=4$、样本标准差、$\epsilon=0.2$、$\beta=0.04$ 和两次更新，是为了完成一个能手算的例子。除非官方资料明确披露，否则不能把这些数字写成 DeepSeek-V3 的真实训练超参数。

### 14.2 最终答案 reward 容易产生投机行为

字符串匹配可能被格式、等价表达式或偶然猜中影响。更稳妥的判分器至少应处理：

- 数值和代数表达式的等价性；
- 分数、小数和单位；
- 集合、区间与多答案格式；
- 答案解析失败时的明确惩罚；
- 必要时的过程验证或程序执行。

### 14.3 MoE 的参数更新不是固定的“37B”

37B 是每个 token 的激活参数量摘要，不是每一步都会被更新的固定参数集合。共享模块、路由器和被选中的专家会共同决定实际梯度落点；不同 token 的专家路径也可能不同。

### 14.4 策略目标不等于真实正确率

GRPO loss 是一个局部代理目标。它可能下降，但独立数学评测不提升；训练题 reward 可能提升，但模型出现答案格式投机、回答变长、KL 漂移或通用能力下降。因此验收至少应同时看：

| 维度 | 要检查什么 |
| --- | --- |
| 能力 | 独立题集正确率，按题型和难度分层 |
| 稳定性 | reward、KL、熵和回答长度曲线 |
| 泛化 | 未见题型、不同语言和不同格式 |
| 保留能力 | 通用知识、代码、指令遵循和安全评测 |
| 成本 | rollout 吞吐、训练 FLOPs、显存和 wall-clock |

## 15. 复现与可证伪扩展

### 15.1 最小复现

不需要从 671B 模型开始。可以用一个小型 causal LM 和整数算术数据集验证下面四件事：

1. 同一个 prompt 采样 $G$ 份回答；
2. 用可执行的答案解析器产生 $r_i$；
3. 检查全对、全错和混合 reward 时的 advantage；
4. 保留 `old_logp`，重复更新同一批回答，观察 ratio、clip 比例和 KL 如何变化。

最小验收条件是：混合 reward 时高分回答的平均 log-probability 倾向上升，低分回答的平均 log-probability 倾向下降；全对或全错组的 reward advantage 为零；把 ratio 改成常数后，策略项不再产生同样的梯度。

### 15.2 一个可证伪的过程监督实验

**假设：** 对同一道题同时提供最终答案 reward 和逐步过程 reward，可以减少错误过程偶然得到正确答案时的策略漂移。

**最小实验：**

- 固定模型、题目、采样温度、组大小和优化步数；
- 对照组只使用 outcome reward；
- 实验组在关键中间步骤加入 verifier；
- 同时评估最终正确率、过程错误率、回答长度和 KL。

**预测：** 如果过程监督真的改善信用分配，实验组应在相同或更低回答长度下减少“错误过程、正确答案”的比例。

**证伪条件：** 实验组只提高训练题 reward，却不提高独立题集正确率，或带来明显的 KL 漂移与通用能力退化。

### 15.3 一个可证伪的 ratio 实验

**假设：** 对同一批 rollout 进行过多次更新，会让当前策略远离旧策略，使 clipped fraction 和 KL 上升，最终削弱训练稳定性。

**最小实验：** 固定 rollout 数据，只改变 `updates_per_rollout`，记录独立评测正确率、clip 比例、KL 和 entropy。

**预测：** 更新次数增加初期可能带来收益，但继续增加会出现 clip 比例和 KL 上升；若正确率不再改善，这说明“在旧样本上继续榨取”已经超过代理目标的有效范围。

## 16. 一句话带走

**GRPO 的核心不是让 reward 对模型反向传播，而是用同题回答的相对得分给 token 级 log-probability 加权，再通过 ratio、clip 和 KL 把这种偏好转成一次受约束的参数更新；要判断它是否真的提升数学能力，最后仍要回到独立评测和清晰的 reward 边界。**

## 参考资料

- [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v2)，DeepSeek-AI，v2，访问日期：2026-10-10。
- [DeepSeek-V3 官方仓库](https://github.com/deepseek-ai/DeepSeek-V3)，模型摘要与训练路线，访问日期：2026-10-10。
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/html/2402.03300v3)，DeepSeek-AI，v3，访问日期：2026-10-10。

