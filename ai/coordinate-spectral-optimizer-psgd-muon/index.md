# 坐标谱优化器：用动量学习 PSGD 坐标，再做部分 Muon 谱校正


Muon 的核心动作是对矩阵动量做谱方向校正；PSGD-Kron 的核心动作是在线学习左右预条件因子。把二者直接相加并不能说明谁贡献了什么，因此本文记录一条更具体的组合路径：**用动量 $M_t$ 拟合 PSGD 的坐标变换，在这个坐标中把原始动量与 Newton–Schulz 谱方向做等范数混合，最后映回参数空间。**

在 51M 参数模型、100,663,296 个 FineWeb-Edu 训练 token 上，正式候选为“$\alpha=0.5$、用 $M$ 拟合 $Q$、前 256 步每步拟合，之后每 8 步拟合一次”。它在三个全新 seed 上都取得低于 Muon 的全量验证 loss，平均差值为 $-0.006636$；代价是平均 wall-clock 高约 7.7%。因此正确的结论是“在这个受控设置下，用额外计算换取稳定但有限的 loss 改善”，而不是“已经普遍超过 Muon”。

本文信息截止到 **2026-09-22**。文中的实验数字来自附带的第二轮算法与 FineWeb 验证记录；它们是当前研究过程中的可复现实验证据，不是已发表论文的 benchmark 结论。

<!--more-->

## 1. 30 秒结论

1. **方法的新增结构是坐标中的谱混合。** PSGD-Kron 学到 $Q_L,Q_R$，先把动量变成 $Z_t=Q_LM_tQ_R^\top$，再在 $Z_t$ 上混合原始方向和五轮 Newton–Schulz 方向。
2. **拟合信号应当是动量而不是单步梯度。** 在开发 seed=1337 上，$M$ 拟合版的全量验证 loss 为 4.114668，优于原始梯度 $G$ 拟合版的 4.134397 和 Muon 的 4.137277。
3. **低频拟合保留了大部分质量。** 前 256 步每步拟合、之后每 8 步拟合的版本是跨 seed 的正式候选；不拟合 $Q$ 的步骤仍使用最新因子更新参数。
4. **在线坐标确实有独立贡献。** 固定 $Q_L=Q_R=I$ 的对照得到 4.395127，明显落后于学习 $Q$ 的版本。
5. **证据边界必须保留。** 三个 seed、一个 51M 模型和一个训练预算不能证明相同 wall-clock 下更快，也不能证明可扩展到更大模型。

## 2. 读完后应能回答

- PSGD-Kron、Muon 和坐标谱混合分别在做什么？
- 为什么用动量 $M_t$ 拟合 $Q$，而不是用原始梯度 $G_t$？
- $Q$ 多久更新一次，参数更新和因子更新是否同步？
- $\alpha=0$、$Q=I$ 等退化情形分别对应什么？
- 三个 seed 的结果能支持什么，不能支持什么？

## 3. 概念与符号

| 符号 / 概念 | 含义 | 本文实验中的边界 |
| --- | --- | --- |
| $W_t$ | 某个隐藏矩阵参数 | 只对矩阵层使用该方向；embedding、bias、normalization 等走 AdamW |
| $G_t$ | global norm clipping 后的梯度 | clipping 阈值为 1 |
| $M_t$ | 一阶动量 | momentum=0.9，Nesterov=False |
| $Q_L,Q_R$ | PSGD-Kron 左右因子 | 按轴大小选择 dense 或 diagonal 结构 |
| $P_L,P_R$ | 对应的预条件矩阵 | $P_L=Q_L^\top Q_L,\ P_R=Q_R^\top Q_R$ |
| $Z_t$ | PSGD 坐标中的动量 | $Z_t=Q_LM_tQ_R^\top$ |
| $\operatorname{NS}_5$ | 五轮 Newton–Schulz 谱变换 | 是有限轮近似，不是精确 SVD |
| $\alpha$ | 谱分支混合强度 | 候选使用 $\alpha=0.5$ |
| full validation | 覆盖约 100 万验证 token 的结果 | 与固定验证窗口分开报告 |
| wall-clock | 一次完整训练的实际耗时 | 不是 FLOPs，也不是达到目标 loss 的时间 |

## 4. 真正的问题：两个优化直觉如何不互相抵消

### 4.1 Muon 在校正什么

对矩阵动量 $M_t$ 做奇异值分解：

$$
M_t=U\Sigma V^\top.
$$

Muon 使用其 polar 方向 $UV^\top$ 的近似，让不同奇异方向的幅度不再完全由当前矩阵的谱值支配。工程实现通常用 Newton–Schulz 迭代近似这一变换，避免每一步都做完整 SVD。

它的优点是方向结构简单、矩阵运算规整；但单步谱校正本身没有显式维护跨步的各向异性统计。

### 4.2 PSGD-Kron 在学习什么

PSGD-Kron 在线维护左右因子：

$$
P_L=Q_L^\top Q_L,\qquad P_R=Q_R^\top Q_R.
$$

如果直接使用 PSGD 方向，更新大致为：

$$
P_LM_tP_R.
$$

这条路径利用了历史统计，但它并不等价于把矩阵谱压平。早期消融中，$\alpha=0$ 的版本固定窗口 loss 为 4.301654，明显差于带谱混合的版本。

### 4.3 坐标谱混合的 insight

组合方法不在原参数坐标中简单相加，而是执行：

```text
动量 M_t
  → PSGD 因子拟合 Q_L, Q_R
  → 变换到坐标 Z_t = Q_L M_t Q_Rᵀ
  → 原始坐标方向与 NS₅(Z_t) 等范数混合
  → 映回参数坐标 D_t = Q_Lᵀ V_t Q_R
  → RMS 归一化与参数更新
```

这样，$Q$ 提供历史统计定义的坐标，NS 分支只在该坐标中进行谱校正。这个结构解释了为什么需要同时做“去掉谱校正”和“固定 $Q=I$”两类对照。

## 5. 方法复原：从梯度到参数更新

以下公式对应当前正式候选；实现中的数值稳定化细节以实验代码为准。

### 5.1 梯度与动量

对全模型梯度做 global norm clipping 后得到 $G_t$。动量为：

$$
\beta_t=\min\left(\frac{t}{t+1},0.9\right),
$$

$$
M_t=\beta_tM_{t-1}+(1-\beta_t)G_t.
$$

由于 $t$ 从 0 开始，第一步等价于 $M_0=G_0$。这里的动量始终保存在原参数坐标中；拟合 $Q$ 时使用的是 $M_t$，不是 $G_t$。

### 5.2 用动量拟合 PSGD 因子

给定当前因子，先加入相对随机阻尼：

$$
F_t=M_t+\left(d+\rho|M_t|\right)\odot\varepsilon_t,
$$

其中 $\varepsilon_{ij}\sim\mathcal N(0,1)$，$d=10^{-9}$，$\rho=2^{-8}$。用更新前的同一组因子计算：

$$
H_t=P_LF_tP_R.
$$

左右轴的二阶统计为：

$$
C_L=H_tH_t^\top,\qquad C_R=H_t^\top H_t.
$$

目标残差因此是：

$$
E_L=C_L-nI_m,\qquad E_R=C_R-mI_n.
$$

这里 $W_t\in\mathbb R^{m\times n}$，所以左轴的目标尺度是输入维度 $n$，右轴的目标尺度是输出维度 $m$。

对任意轴 $a\in\{L,R\}$，令 $k_L=n,\ k_R=m$，并使用随机子空间幂迭代估计谱范数：

$$
b_a=\widehat{\lVert C_a\rVert}_2+k_a,
$$

$$
\ell_a^+=\max\{0.9\ell_a+0.1b_a,\ b_a\}.
$$

稠密因子的主更新为：

$$
\bar Q_a=
\left[I-\frac{\eta_Q}{\ell_a^+}(C_a-k_aI)\right]Q_a,
\qquad \eta_Q=0.1.
$$

若某个轴使用对角因子 $q_a$，则将完整 Gram 矩阵替换为另一轴上的平方和 $c_i$：

$$
q_{a,i}^+=q_{a,i}
\left[1-\frac{\eta_Q}{\ell_a^+}(c_i-k_a)\right].
$$

实际代码还会对稠密 $Q$ 做在线 Procrustes 稳定化；上式展示的是主要 PSGD 更新，而不是逐行等同于全部数值实现。

### 5.3 因子结构与更新频率

轴大小为 $d_a$ 时，当

$$
1<d_a\le1024,\qquad d_a^2\le2mn
$$

使用 dense 因子，否则使用 diagonal 因子。

动量每一步都更新，但 $Q_L,Q_R$ 不必每一步重算。正式候选采用：

$$
\text{前 256 步每步拟合 }Q；\qquad
\text{之后每 8 步拟合一次 }Q.
$$

没有拟合 $Q$ 的步骤直接沿用最近因子，但仍用当前 $M_t$ 计算参数方向。这个区别很重要：低频的是坐标统计更新，不是把整个优化器冻结 8 步。

### 5.4 在坐标中做谱校正

使用当前可用因子，把动量变换到 PSGD 坐标：

$$
Z_t=Q_LM_tQ_R^\top.
$$

归一化后进行五轮 Newton–Schulz：

$$
X_0=\frac{Z_t}{\lVert Z_t\rVert_F+10^{-7}},
$$

$$
A_k=X_kX_k^\top,
$$

$$
X_{k+1}
=3.4445X_k+
\left(-4.775A_k+2.0315A_k^2\right)X_k.
$$

执行五轮后记为：

$$
S_t=\operatorname{NS}_5(Z_t).
$$

它是有限轮 polar 近似，不应被描述成精确 SVD 或“严格把所有非零奇异值变成 1”。

为了让谱分支与原始分支在同一个幅度尺度比较，先做 Frobenius 对齐：

$$
\widetilde S_t=
\begin{cases}
\dfrac{\lVert Z_t\rVert_F}{\lVert S_t\rVert_F}S_t,
&\lVert S_t\rVert_F>0,\\
0,&\lVert S_t\rVert_F=0.
\end{cases}
$$

然后混合：

$$
V_t=(1-\alpha)Z_t+\alpha\widetilde S_t.
$$

当前候选使用 $\alpha=0.5$。这不是“PSGD 更新和 Muon 更新各乘 0.5”，而是**在 PSGD 坐标中**把原始动量方向与谱校正方向做等权混合。

### 5.5 映回参数坐标并归一化

先映回原参数坐标：

$$
D_t=Q_L^\top V_tQ_R.
$$

定义：

$$
\operatorname{RMS}(D_t)=\frac{\lVert D_t\rVert_F}{\sqrt{mn}}.
$$

最终方向的 RMS 固定为 $1/\sqrt n$：

$$
\widehat D_t=
\frac{D_t}{\sqrt n\,\operatorname{RMS}(D_t)}
=\frac{\sqrt m}{\lVert D_t\rVert_F}D_t.
$$

隐藏矩阵的参数更新为：

$$
W_{t+1}=(1-\eta_t\lambda)W_t-\eta_t\widehat D_t.
$$

当前实验中 $\lambda=0$，因此就是 $W_{t+1}=W_t-\eta_t\widehat D_t$。矩阵峰值学习率为 0.01；embedding、输出头、bias 和 normalization 等辅助参数使用 AdamW，峰值学习率为 $6\times10^{-4}$。

## 6. 一行公式与两个退化情况

定义坐标中的谱混合：

$$
\Phi_\alpha(Z)
=(1-\alpha)Z+
\alpha\frac{\lVert Z\rVert_F}{\lVert\operatorname{NS}_5(Z)\rVert_F}
\operatorname{NS}_5(Z),
$$

以及 RMS 归一化：

$$
\mathcal N_n(D)=\frac{\sqrt m}{\lVert D\rVert_F}D.
$$

完整候选可以写成：

$$
\boxed{
W_{t+1}=W_t-\eta_t\mathcal N_n
\left[
Q_L^\top\Phi_{0.5}
\left(Q_LM_tQ_R^\top\right)Q_R
\right]
}
$$

其中 $Q_L,Q_R$ 由动量在线拟合，前 256 步每步更新，之后每 8 步更新一次。

当 $\alpha=0$ 时：

$$
D_t=Q_L^\top Q_LM_tQ_R^\top Q_R=P_LM_tP_R,
$$

即带最终 RMS 归一化的 PSGD-Kron 方向。

当 $Q_L=I,\ Q_R=I,\ \alpha=1$ 时：

$$
D_t=\operatorname{NS}_5(M_t),
$$

它在方向结构上退化为 Muon 式谱更新。但实际 Muon 基线使用 momentum=0.95、Nesterov=True 等不同配方，因此不能把这个退化情形直接称为“完整 Muon”。

## 7. 实验证据：每个对照只回答一个问题

实验模型为 8 层、8 heads、hidden size 512、context 512，约 51.17M 参数；每步 32,768 token，训练 3,072 updates，总训练量为 100,663,296 token，数据为固定 FineWeb-Edu 子集，使用 GPT-2 BPE，硬件为单张 A800 80GB。除非特别说明，结果是单 seed 开发证据。

### 7.1 小型 Shakespeare：先筛选 $\alpha$

6 层、6 heads、宽度 384、context 256、batch 64、训练 1,000 步的字符级实验中：

| Matrix LR | $\alpha=0$ | $\alpha=0.5$ | $\alpha=1$ |
| ---: | ---: | ---: | ---: |
| 0.002 | 1.615846 | **1.611920** | 1.640968 |
| 0.01 | **1.491737** | 1.492434 | 1.501545 |
| 0.02 | 1.494682 | **1.486598** | 1.491019 |

Muon 对照为 1.482030。这个实验只能说明 $\alpha=0.5$ 值得进入下一轮，不能说明它已经超过 Muon；小型字符数据也不足以代表语言模型预训练。

### 7.2 FineWeb 第一轮：原始梯度拟合 Q

固定验证窗口结果为：

| 方法 | 固定窗口 val | 总耗时 |
| --- | ---: | ---: |
| AdamW | 4.189980 | 533.46 s |
| Muon | 4.116105 | 606.47 s |
| 坐标谱，$G$ 拟合、每步更新 $Q$ | **4.111403** | 736.29 s |

扩展到全部验证窗口后：

| 方法 | 全量验证 loss |
| --- | ---: |
| AdamW | 4.212151 |
| Muon | 4.137277 |
| 坐标谱，$G$ 拟合 | **4.134397** |

差值只有 $-0.002880$，而完整耗时比 Muon 高约 21%。这支持“组合方向有潜力”，但不支持“质量改善已经稳健”。

### 7.3 去掉谱校正：$\alpha=0$

在相同开发设置中令 $\alpha=0$ 并跳过 NS5，固定窗口 loss 为 4.301654，明显差于原 $\alpha=0.5$ 的 4.111403。

【来源结论】在该模型、数据和学习率下，谱校正对质量有实质贡献。

【不能外推】这不是对所有学习率、所有预条件器或所有模型规模的普遍证明。

### 7.4 用动量拟合 Q：从 G 到 M

保持 $\alpha=0.5$，只改变拟合 $Q$ 的信号：

| 配方 | 全量验证 loss |
| --- | ---: |
| $G$ 拟合、每步 Q | 4.134397 |
| **$M$ 拟合、每步 Q** | **4.114668** |
| Muon | 4.137277 |

开发 seed=1337 上，$M$ 拟合版相对 Muon 改善：

$$
4.137277-4.114668=0.022609.
$$

若把验证 loss 差异换算成几何平均 perplexity 的相对变化，约为：

$$
1-e^{-0.022609}\approx2.24\%.
$$

这个数字只适用于该开发 seed 和该验证定义；它不能被写成通用 perplexity 提升。

### 7.5 低频拟合：质量与开销的折中

将 $M$ 拟合 Q 改为“前 256 步每步、之后每 8 步”：

| 配方 | 固定窗口 val | 全量 val |
| --- | ---: | ---: |
| $M$，每步拟合 Q | **4.094941** | **4.114668** |
| $M$，每 8 步拟合 Q | 4.101244 | 4.125453 |

低频版损失了一小部分开发质量，但将相对 Muon 的完整运行时间开销从约 21% 降到约 7.7% 的量级，因此被选为正式三 seed 候选。一次运行曾与误启动的进程并行，不能用其完整 wall time 做速度结论；中位 step 只作参考。

### 7.6 固定 Q=I：在线坐标是否真的有用

保留 $\alpha=0.5$ 的谱混合，但固定：

$$
Q_L=I,\qquad Q_R=I.
$$

全量验证 loss 为 4.395127，明显落后于学习 Q 的 4.114668。

【来源结论】在当前配方下，收益不只是“动量 + NS5”的结果，在线学习的 Q 因子具有独立贡献。

【仍未证明】这个贡献一定来自 Q 的非对角元素；要回答这个问题，还需要 diagonal-Q 对照。

## 8. 三个独立 seed 的正式验证

正式候选固定为：

$$
\alpha=0.5,\quad
\text{用 }M\text{ 拟合 }Q,\quad
\text{前 256 步每步拟合，之后每 8 步拟合}.
$$

三个全新 seed 的全量验证结果：

| seed | 坐标谱方法 | Muon | 配对差值 |
| ---: | ---: | ---: | ---: |
| 2027 | **4.130653** | 4.137272 | -0.006619 |
| 2028 | **4.126988** | 4.131270 | -0.004282 |
| 2029 | **4.127468** | 4.136476 | -0.009008 |
| **平均** | **4.128370** | **4.135006** | **-0.006636** |

三个 seed 的改善方向一致。将平均 loss 差换算为几何平均 perplexity 的相对变化：

$$
1-e^{-0.006636}\approx0.66\%.
$$

配对差值的样本标准差约为 0.002363。样本数仍然很小，所以更稳妥的表达是“在三次配对复现实验中方向一致”，而不是给出强统计显著性的结论。

### 8.1 计算成本

| seed | 坐标谱方法 | Muon |
| ---: | ---: | ---: |
| 2027 | 649.64 s | 602.47 s |
| 2028 | 652.60 s | 604.61 s |
| 2029 | 650.50 s | 605.99 s |
| **平均** | **650.91 s** | **604.36 s** |

平均额外完整运行时间为：

$$
\frac{650.91}{604.36}-1\approx7.70\%.
$$

所以当前证据支持的完整句子是：

> 在相同训练 token 下，坐标谱候选用约 7.7% 的额外训练时间换取约 0.006636 的平均验证 loss 改善，且三个独立 seed 的改善方向一致。

如果比较相同 wall-clock，候选至少需要少走：

$$
1-\frac{604.36}{650.91}\approx7.15\%
$$

的训练步数，才能在训练时间意义上抵消慢速。目前没有这样的证据。

## 9. Claim–evidence map

| Claim | 直接证据 | 强度 | 不能外推成什么 |
| --- | --- | --- | --- |
| 在 PSGD 坐标中混合谱方向有用 | $\alpha=0$ 消融与 $\alpha=0.5$ 对照 | 中 | 不能证明所有模型都偏好 $\alpha=0.5$ |
| 用动量拟合 Q 优于用 G | seed=1337 的配对开发实验 | 中 | 不能证明跨 seed、跨数据集仍保持同样差值 |
| 在线 Q 有独立贡献 | 固定 $Q=I$ 对照 | 中 | 不能区分 dense 与 diagonal 非对角结构的贡献 |
| 低频 Q 拟合可以保留质量 | 每步与每 8 步的开发对照 | 中 | 不能由一次运行推导最优更新间隔 |
| 正式候选在三 seed 上优于 Muon | 2027–2029 配对全量验证 | 中高 | 不能证明相同 wall-clock 更优或规模可扩展 |
| 候选有额外计算成本 | 三 seed 完整运行时间 | 中高 | 不能把 A800 上的比例当成所有硬件常数 |

## 10. 局部数学性质与边界

冻结 $Q_L,Q_R$ 时，令 $Z_t=Q_LM_tQ_R^\top$，则：

$$
\langle M_t,D_t\rangle=\langle Z_t,V_t\rangle.
$$

若谱分支使用精确 polar，便有：

$$
\langle Z_t,\operatorname{polar}(Z_t)\rangle=\lVert Z_t\rVert_* > 0.
$$

因此混合方向与动量的配对为正。有限轮 NS5 的单轮奇异值乘子为：

$$
p(s^2)=3.4445-4.775s^2+2.0315s^4.
$$

该多项式的全局最小值约为：

$$
3.4445-\frac{4.775^2}{4\times2.0315}\approx0.638615>0.
$$

在精确算术下，NS5 不会把一个奇异方向翻转成反方向。

这只是冻结坐标下的局部方向性质，不是非凸训练的全局收敛证明，也不保证方向与当前梯度 $G_t$ 始终正内积。真实训练中 $Q$ 会更新、存在有限精度和 RMS 归一化，不能把这段推导当成完整优化理论。

## 11. 复现计划

最小复现实验应固定以下条件，而不是只复制一个优化器类名：

1. 固定 8 层、hidden size 512、context 512、每步 32,768 token 和 3,072 updates。
2. 固定 FineWeb-Edu 子集、GPT-2 BPE、验证窗口定义以及单张 A800 80GB 环境。
3. 同时运行配对的 Muon 与候选版本，并保存每个 seed 的 full validation、step time 和完整 wall-clock。
4. 将“每步拟合 Q”和“低频拟合 Q”作为两个不同配置，不把质量最好版和证据最完整版混写。
5. 额外记录 Q 的 dense/diagonal 选择、NS5 的范数、因子更新耗时和随机阻尼种子。

### 可证伪扩展

| 假设 | 最小实验 | 预期支持 | 证伪条件 |
| --- | --- | --- | --- |
| Q 的非对角结构贡献了主要收益 | 固定相同统计，改用 diagonal-only Q | diagonal-only 接近 dense-Q | diagonal-only 与 $Q=I$ 一样，或 dense 优势消失 |
| 更低频的 Q 更新仍可保留质量 | 比较间隔 4、8、16、32，配对三个 seed | loss 平台而开销下降 | 间隔稍增即显著退化 |
| 额外开销能换来更快达到目标 loss | 按 wall-clock 重采样训练曲线 | 候选更早达到同一 loss | 候选始终慢于 Muon |
| $\alpha=0.5$ 不是开发集偶然点 | 在新 seed 和新验证窗口扫描 $\alpha$ | 最优区间仍在中间值 | 新设置偏好 $\alpha=0$ 或 $\alpha=1$ |

## 12. 一句话带走

**这是一种“先用动量学习坐标，再在坐标里做部分谱校正”的优化器；目前最可靠的证据是三 seed、同 token 数下平均比 Muon 低 0.006636，但它仍以约 7.7% 的额外训练时间为代价，尚未证明相同 wall-clock 或更大规模下仍占优。**

## 参考资料

- [Online Second Order Methods for Non-Convex Stochastic Optimizations](https://arxiv.org/abs/1803.09383)，PSGD 的原始方法论文。
- [psgd_torch](https://github.com/lixilinx/psgd_torch)，PSGD-Kron 的 PyTorch 实现与工程参考。
- [Preconditioning Benefits of Spectral Orthogonalization in Muon](https://arxiv.org/abs/2601.13474)，Muon 谱正交化的理论分析。
- [Practical Efficiency of Muon for Pretraining](https://arxiv.org/abs/2505.02222)，Muon 预训练效率研究。
- [FineWeb-Edu 数据集卡](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)，数据集说明与版本信息。
- 本文实验记录：`psgd_coordinate_spectral_algorithm_round2.md` 与 `fineweb_followup_optimization_round2.md`，信息截止 2026-09-22。

