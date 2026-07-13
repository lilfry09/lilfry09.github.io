# R3 和异步 RL：强异步 MoE 训练中的 routing replay 问题


### **R3 和异步 RL 解决的是两个正交问题：异步导致的是“用旧 policy 的数据更新新 policy”，R3 修的是“对同一个 token、同一组参数，训练重算时没有复现 rollout 当时的 MoE routing”。即便 Composer 天生允许 policy staleness，也仍需先保证行为 policy 的 token probability 能被准确重建，否则 off-policy correction、KL 和 policy ratio 都会被错误 routing 污染。**

关于这件事，有两个容易混在一起的判断：一个是“逻辑上不等价”的区分，这个分类基本正确；另一个是“想更新生成 token 时参与计算的 experts”，它也抓到了现象，但最后一句需要更精确一点：R3 的关键未必是**冻结并只更新原 experts**，而是**重放 rollout routing，以便正确重建 behavior-policy forward 或训练所需的概率和梯度路径**。两件事很接近，却不能完全画等号。

<!--more-->

### **先拆开三种“不一致”**

这里至少有三类 mismatch。混在一起聊，很快就会像三根耳机线放口袋里一样自动打结。

#### **1. Policy staleness：异步 RL 主动接受的不一致**

token 由较旧的 rollout policy $\mu=\pi_{\theta_b}$ 产生，但进入训练时，learner 已更新到当前 policy $\pi_{\theta}$：

$$
a_t \sim \pi_{\theta_b}(\cdot\mid s_t),
\qquad
\theta_b \neq \theta
$$

这是**算法层面的不等价**。异步 RL 知道数据来自旧 policy，并通过 version control、importance sampling、PPO clipping、KL constraint、丢弃过旧样本等方法处理。

它无法被 R3 消除。模型权重已经不同，重放相同 routing 也不会把 $\theta$ 变回 $\theta_b$。

#### **2. Train–inference numerical mismatch：同一 policy 本应等价却没有等价**

即使权重 snapshot 完全相同，rollout engine 和 training engine 也可能因为：

- kernel 不同；
- tensor/expert parallel 切分不同；
- precision 不同；
- quantization 不同；
- sequence packing 不同；
- attention 实现不同；
- MoE router 的 top-$k$ tie breaking 不同；

而计算出略有差异的 logits。

这是**实现层面的偏差**。理论上双方想表示同一个 policy，实际 forward 却不完全一致。

#### **3. MoE routing replay mismatch：同一 token 经过了不同 experts**

对 MoE token $x_t$，rollout 时 router 选择：

$$
\mathcal E_t^{\mathrm{roll}}
=
\operatorname{TopK}\bigl(g_{\theta_b}^{\mathrm{roll}}(x_t)\bigr)
$$

训练重算时，如果重新运行 router，可能得到：

$$
\mathcal E_t^{\mathrm{train}}
=
\operatorname{TopK}\bigl(g_{\theta}^{\mathrm{train}}(x_t)\bigr)
$$

两者不同可能有两个来源：

1. **同版本权重下的执行差异**：数值扰动、并行布局、capacity、token ordering 导致 routing 改变；
2. **异步造成的参数差异**：router 权重和 hidden states 已变化，当前 policy 本来就会选不同 experts。

R3 主要针对第一类，并可能为第二类中的“准确重建 behavior forward”提供支撑；但它不能把旧 policy 与当前 policy 在逻辑上变成同一个 policy。

---

### **为什么强异步系统反而更需要 R3**

同步系统里，rollout 结束后立即训练，$\theta_b$ 与 $\theta$ 通常比较接近。异步系统中，样本在 queue 里待一段时间，可能跨过多个 learner updates。此时正确的 PPO ratio 应该是：

$$
\rho_t
=
\frac{
\pi_{\theta}(a_t\mid s_t)
}{
\pi_{\theta_b}(a_t\mid s_t)
}
$$

这里分母必须是**真正生成该 token 的 behavior policy probability**。如果 rollout 时 token 经过 experts $\{E_2,E_7\}$，但训练侧为了计算 old logprob 又重新路由到了 $\{E_3,E_7\}$，你得到的就不是：

$$
\log \pi_{\theta_b}^{\mathrm{actual}}(a_t\mid s_t)
$$

而是某个训练侧重构出来的近似：

$$
\log \widetilde{\pi}_{\theta_b}(a_t\mid s_t)
$$

于是实际使用的 ratio 变成：

$$
\widetilde{\rho}_t
=
\frac{
\pi_{\theta}(a_t\mid s_t)
}{
\widetilde{\pi}_{\theta_b}(a_t\mid s_t)
}
$$

现在 ratio 同时混入了两种变化：

$$
\log \widetilde{\rho}_t
=
\underbrace{
\log \pi_{\theta}(a_t\mid s_t)
-
\log \pi_{\theta_b}^{\mathrm{actual}}(a_t\mid s_t)
}_{\text{真实 policy drift}}
+
\underbrace{
\log \pi_{\theta_b}^{\mathrm{actual}}(a_t\mid s_t)
-
\log \widetilde{\pi}_{\theta_b}(a_t\mid s_t)
}_{\text{训推/routing 重构误差}}
$$

异步 RL 的 correction 只想处理第一项；第二项是脏噪声。policy lag 越大，importance ratio 本来就越容易极端，这时再叠加 routing error，会更容易触发 PPO clipping、造成错误 KL 估计或扭曲 advantage weighting。

所以不是“既然异步本来就不等价，那训推一致性无所谓”。恰恰相反：

> **只有先消除实现层面的假不一致，算法才能看清真正的 policy staleness 有多大。**

这和测量运动中的目标类似：目标真的在动是一回事，相机自己抖是另一回事。不能因为目标在动，就宣布防抖没有意义。

---

### **R3 大概率在重放什么**

R3 可以理解为记录 rollout 时每个 token 的 MoE routing 信息，并在训练或 logprob 重算时 replay。典型信息可能包括：

- selected expert IDs；
- top-$k$ expert order；
- router weights/gates；
- token-to-expert dispatch；
- capacity overflow 或 dropped-token 状态；
- 必要时的 parallel placement metadata。

这样，对一条 rollout trajectory，训练侧能区分两个问题：

#### **Behavior-policy reconstruction**

需要知道 rollout 当时到底计算了什么。若系统保存了 rollout 时的 old logprobs，分母可以直接使用，不必重算；但 reference KL、质量检查、某些 loss 结构，以及 old-logprob validation 仍可能需要可靠 replay。

#### **Current-policy evaluation**

PPO/GRPO 的分子通常应是当前 policy 对已采样 token 的 probability。此时是否也强制当前 policy 沿旧 routing，要看 R3 的具体 objective 定义，这里非常关键。

若强制当前 policy 使用 rollout routing，计算的是：

$$
\pi_{\theta}
\left(
a_t\mid s_t,\mathcal E_t^{\mathrm{roll}}
\right)
$$

而不是 current policy 自由路由得到的：

$$
\pi_{\theta}
\left(
a_t\mid s_t,\mathcal E_t^{\mathrm{current}}
\right)
$$

前者更稳定，也让梯度流经实际参与 rollout 的 experts；后者更忠实于部署时当前 policy 的真实行为。两者各有道理，但优化对象不同。不能只说“replay routing”就跳过这一层。

---

### **“应该更新产生 token 的 experts”对不对？**

**作为 credit assignment 直觉，它基本对；作为完整的 policy-gradient 结论，还少了一半。**

假设 rollout 时 token 经由 experts 2 和 7 产生。固定 routing path 重算 loss 时，token-level gradient 会流向：

- shared attention 和 dense layers；
- experts 2 和 7；
- output head；
- 可能还有 router/gating weights。

如果训练时重新路由成 experts 3 和 7，那么 expert 2 不再收到该 token 的梯度，expert 3 却在为一个它没有参与生成的行为承担更新。这就是 expert credit mismatch。

但 router 本身也是 policy 的一部分。若完全把旧 routing 当常量，可能发生两个问题：

1. selected expert 得到稳定 credit，但 router 未必得到正确的学习信号；
2. current policy 真实部署时可能会改走另一条 path，而训练 objective 只优化旧 path。

更准确的拆法是：

$$
\pi_\theta(a_t\mid s_t)
=
\sum_z
p_\theta(z\mid s_t)\,
p_\theta(a_t\mid s_t,z)
$$

其中 $z$ 是 latent routing decision。实际 top-$k$ MoE 通常没有精确求和，而是稀疏选出少量 experts。rollout 记录的是某个 $z_b$。重放 $z_b$ 优化的是近似条件概率：

$$
p_\theta(a_t\mid s_t,z_b)
$$

而完整 policy 还包含 router probability：

$$
p_\theta(z_b\mid s_t)
$$

因此最原则化的方案可能是：

- **expert path** 使用 rollout routing replay，保证 token-to-expert credit；
- **router objective** 单独处理，例如记录 rollout router logits/gates，计算 router-level ratio、KL 或辅助 loss；
- **current free-routing forward** 额外用于监测实际 policy drift。

这比“训练时一律强制旧 routing”更完整。

---

### **强异步下，应该 replay 旧 routing 还是用当前 routing？**

取决于这次 forward 要回答什么问题。

| 目的 | 更合理的 routing |
|---|---|
| 重建 behavior policy 的 old logprob | rollout 时的 routing |
| 验证 rollout engine 与 train engine 是否一致 | rollout 时的 routing |
| 给产生该 token 的 experts 分配 credit | rollout 时的 routing |
| 评估当前部署 policy 的真实分布 | 当前 policy 自由 routing |
| 计算严格的 current/behavior ratio | 理想上需要两套各自真实的 routing |
| 稳定训练的 surrogate objective | 可能使用 replay routing，但需承认它是 surrogate |

理想的 off-policy ratio 应是：

$$
\rho_t
=
\frac{
\pi_{\theta}^{\mathrm{free-route}}(a_t\mid s_t)
}{
\pi_{\theta_b}^{\mathrm{roll-route}}(a_t\mid s_t)
}
$$

这里分子按当前 policy 自由 routing，分母按行为 policy 的实际 routing。

但这种做法可能导致分子和分母经过不同 experts，ratio 方差很大，梯度也主要更新 current-route experts，而不是 behavior-route experts。为了稳定性，系统可能使用 path-conditioned surrogate：

$$
\rho_t^{\mathrm{route}}
=
\frac{
\pi_{\theta}(a_t\mid s_t,z_b)
}{
\pi_{\theta_b}(a_t\mid s_t,z_b)
}
$$

它回答的是：

> 在固定旧 routing path 的条件下，当前参数相对行为参数把该 token 的概率改了多少？

这通常更稳定，也更符合“更新参与 rollout 的 experts”的工程直觉；但它不完全等价于整个自由路由 MoE policy 的 importance ratio。R3 若采用该路线，本质上是在 bias 与 variance、真实 current policy 与稳定 credit assignment 之间做选择。

---

### **为什么异步会放大 MoE routing 的麻烦**

Dense model 的权重变了，前向结果平滑变化的可能性较大。MoE routing 中有一个离散 top-$k$ 边界。两个 expert 得分只差一点时，微小参数变化就可能让路径突然跳变：

$$
g_2(x)=0.501,\quad g_3(x)=0.499
$$

更新后可能变为：

$$
g_2(x)=0.498,\quad g_3(x)=0.502
$$

logits 变化很小，实际执行图却从 expert 2 跳到 expert 3。异步 policy lag 让这种 path flip 更频繁。

因此 MoE 异步 RL 有两层 drift：

$$
\text{policy drift}
=
\text{within-path parameter drift}
+
\text{routing-path drift}
$$

后者是离散的，不能总被普通 token-level KL 平滑地捕捉。两个模型可能最终 token distribution 暂时接近，但内部 expert allocation 已大幅变化；也可能只因 routing flip 就出现突然的 logprob 跳变。

R3 的价值之一就是把这两类变化解耦：

- 固定 routing 后，测 within-path drift；
- 自由 routing 与 replay routing 对比，测 routing-induced drift。

这对于 Composer 这类强异步系统尤其重要，因为它需要知道样本变旧究竟是“参数稍旧”，还是“已经跨过大量 routing boundaries”。

---

### **几个常见判断的边界**

#### **“R3 fix 的是训推不一致，异步 RL 也存在这个问题吧”**

对。异步 RL 不仅存在，还可能更严重。policy staleness 不会覆盖训推不一致；二者会叠加。

#### **“R3 解决的是逻辑等价但实际执行不等价；异步 RL 逻辑上就不等价”**

也对。这是最清楚的分类：

- 同 snapshot 下 routing 不同：本应等价却执行不等价；
- 不同 snapshot：算法上允许的不等价。

需要补一句：即使不同 snapshot，仍须准确重建每个 snapshot 当时的实际执行，才能做正确的 off-policy accounting。

#### **“RL 里要求 MoE 训练完全遵循 rollout routing path 才最好？”**

不能直接说“完全遵循就一定最好”。更稳妥的说法是：

- 重建 behavior probability 和做 expert credit assignment 时，replay rollout path 很有价值；
- 表示当前 policy 的真实行为时，应允许当前 router 自由选择；
- 最好的训练方式可能是双 forward 或分解 objective，而不是永远锁死旧 path。

#### **“不用 replay，训练时更新的 experts 和推理时实际运转的 experts mismatch”**

现象判断正确。不过若训练计算的是 current-policy free-routing loss，那些新 experts 并非“错误 experts”，而是当前 policy 真会使用的 experts。真正的问题是，这个 loss 是否仍能为 behavior trajectory 提供低方差、语义清晰的 credit。R3 选择 replay，通常是在加强 causal credit continuity，而不只是修一个程序 bug。

---

### **一个更准确的整体理解**

Composer 的强异步 RL 可以同时维护三样东西：

1. **旧 policy 的事实记录**：policy version、old logprobs、rollout routing；
2. **当前 policy 的真实评估**：当前权重、自由 routing、current logprobs；
3. **稳定的训练 surrogate**：固定旧 routing 后计算 path-conditioned ratio 和 expert gradients。

相应地，可以监控三种量：

$$
\mathrm{KL}_{\mathrm{total}}
=
\mathrm{KL}
\left(
\pi_{\theta_b}^{\mathrm{roll}},
\pi_\theta^{\mathrm{free}}
\right)
$$

$$
\mathrm{KL}_{\mathrm{within\ path}}
=
\mathrm{KL}
\left(
\pi_{\theta_b}(\cdot\mid z_b),
\pi_\theta(\cdot\mid z_b)
\right)
$$

以及 routing change rate：

$$
P\left(
z_\theta\neq z_b
\right)
$$

这三项分别对应整体 policy staleness、固定路径下的参数变化，以及 router path drift。只看普通 KL 或只看 policy version，都不足以描述强异步 MoE 的真实 off-policy 程度。

因此，最简洁的结论是：

> **Composer 用 R3，不是为了让异步数据重新变成 on-policy，而是为了让“旧到什么程度”可测、让 behavior policy 可重建，并让 rollout token 的 expert credit 不被训练侧重新 routing 篡改。异步解决执行流水线，R3解决执行语义的可追溯性；前者越激进，后者通常越重要。**

