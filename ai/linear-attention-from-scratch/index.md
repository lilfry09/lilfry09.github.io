# 线性注意力从头到尾讲清楚：从注意力矩阵到可递推状态


> **Linear Attention 的目标是避免显式构造 $n\times n$ 的注意力矩阵，而是把“每个 query 与所有 key 的交互”改写成“query 读取一个累积状态”。**

这句话比“Linear Attention 把复杂度从平方降到线性”更接近本质。复杂度下降只是结果，真正发生的结构变化是：标准 Attention 显式维护 token 与 token 之间的关系，而 Linear Attention 先把所有 key–value 信息写进一个状态，再让 query 从状态中读出结果。

这篇文章从标准 Attention 出发，把这次改写完整推一遍。读完后，你应该能回答四个问题：为什么不能直接先算 $K^\top V$；归一化需要保存什么额外状态；因果 Linear Attention 为什么像 RNN；以及它用固定状态换走 $n\times n$ 矩阵时究竟牺牲了什么。

<!--more-->

## 如果只看三句话

1. 标准 Attention 是“每个 query 逐个匹配所有 key”，因此会产生 $n\times n$ 的注意力矩阵。
2. 如果相似度可以分解为 $\operatorname{sim}(q,k)=\phi(q)^\top\phi(k)$，就能利用结合律先计算 $\sum_j\phi(k_j)v_j^\top$，再由 query 读取。
3. 在因果场景中，这个汇总量可以逐 token 更新，于是 Attention 变成了一个具有固定大小状态的递推模型。

最值得记住的图景是：

$$\boxed{\text{显式 token-to-token 注意力图}\quad\longrightarrow\quad\text{可累积的记忆状态}}$$

## 1. 先把标准 attention 写清楚

标准 scaled dot-product attention 是：

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

其中：

- `Q \in \mathbb{R}^{n \times d_k}`
- `K \in \mathbb{R}^{n \times d_k}`
- `V \in \mathbb{R}^{n \times d_v}`
- `n` 是序列长度

对第 `i` 个 token 来说，它的输出可以写成：

$$
o_i = \sum_{j=1}^{n}\alpha_{ij} v_j
$$

其中权重是：

$$
\alpha_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d_k})}{\sum_{l=1}^{n}\exp(q_i^\top k_l / \sqrt{d_k})}
$$

这就是 attention 的核心：`q_i` 和所有 `k_j` 做匹配，然后按匹配结果对 `v_j` 加权求和。

问题也在这里。

因为每个 `q_i` 都要和所有 `k_j` 比一遍，所以你会得到一个 `n \times n` 的注意力矩阵：

$$
QK^\top \in \mathbb{R}^{n \times n}
$$

所以长度维度上的代价是二次的。序列一长，时间和显存就都顶不住。

## 2. 为什么不能直接先算 `K^T V`

如果没有 softmax，attention 其实就是普通矩阵乘法：

$$
O = QK^\top V
$$

这时由于矩阵乘法满足结合律，可以改写成：

$$
O = Q(K^\top V)
$$

这样一来，先算 `K^\top V`，就不需要显式构造 `n \times n` 的中间矩阵了。

但标准 attention 不是这个式子，而是：

$$
O = \mathrm{softmax}(QK^\top)V
$$

这里的 softmax 是按行做归一化的。它把每一行的打分变成一个概率分布，所以每个 `o_i` 都依赖于“这一行所有 token 的相对关系”。

这意味着：

$$
\mathrm{softmax}(QK^\top)V \neq Q(K^\top V)
$$

原因很简单：`softmax` 不是线性的，也不能从中间那一步直接拆出去。  
所以“先算 `K^T V`”这个想法，只有在没有 softmax 时才成立。

## 3. linear attention 的真正思路：让相似度可以分解

如果你想把 attention 变成线性复杂度，最自然的思路不是去硬拆矩阵乘法，而是想办法把注意力写成某种可分解形式。

更一般地，可以把归一化 attention 写成：

$$
o_i=\frac{\sum_{j=1}^{n}\operatorname{sim}(q_i,k_j)v_j}{\sum_{j=1}^{n}\operatorname{sim}(q_i,k_j)}
$$

如果相似度可以写成某个特征映射的内积：

$$
\operatorname{sim}(q,k)=\phi(q)^\top\phi(k)
$$

这样一来，注意力权重就不再是必须显式计算的两两匹配，而可以变成“先聚合 `K` 和 `V`，再让 `Q` 去读这个聚合结果”。

为了看清楚，先看分子：

$$
\sum_{j=1}^{n}\operatorname{sim}(q_i,k_j)v_j
$$

代入可分解相似度：

$$
\sum_{j=1}^{n}\phi(q_i)^\top \phi(k_j) v_j
$$

由于 `\phi(q_i)` 和求和变量 `j` 无关，可以把它提到外面：

$$
\phi(q_i)^\top \sum_{j=1}^{n}\phi(k_j)v_j
$$

这一步就是 linear attention 的核心。分母也可以用同样的方法拆开，并用另一个累积量完成归一化。

注意，这里不是把 `K^T V` 原封不动提前算掉，而是把“核化后的 key/value 聚合”提前算掉。

这里有一条很重要的概念边界：有些 Linear Attention 直接选择新的可分解核，例如 `ELU(x)+1`；另一些方法才是去近似 softmax 的指数点积核。前者重新定义了注意力相似度，后者试图逼近原来的 softmax attention，不能笼统地说成同一件事。

## 4. 详细推导：从逐 token 公式到可实现形式

我们从单个位置 `i` 开始。

标准 softmax attention 的输出是：

$$
o_i = \sum_{j=1}^{n}\frac{\exp(q_i^\top k_j)}{\sum_{l=1}^{n}\exp(q_i^\top k_l)}v_j
$$

把分子分母分开写：

$$
o_i = \frac{\sum_{j=1}^{n}\exp(q_i^\top k_j)v_j}{\sum_{l=1}^{n}\exp(q_i^\top k_l)}
$$

现在引入特征映射 `\phi(\cdot)`，近似：

$$
\exp(q_i^\top k_j)\approx \phi(q_i)^\top\phi(k_j)
$$

于是分子变成：

$$
\sum_{j=1}^{n}\phi(q_i)^\top\phi(k_j)v_j
$$

把 `\phi(q_i)` 提到外面：

$$
\phi(q_i)^\top\left(\sum_{j=1}^{n}\phi(k_j)v_j^\top\right)
$$

这里为了维度一致，把 `v_j` 写成列向量时，里面是一个外积累积。  
定义：

$$
S=\sum_{j=1}^{n}\phi(k_j)v_j^\top
$$

那么分子就是：

$$
\phi(q_i)^\top S
$$

分母同理：

$$
\sum_{j=1}^{n}\phi(q_i)^\top\phi(k_j)
=
\phi(q_i)^\top\left(\sum_{j=1}^{n}\phi(k_j)\right)
$$

定义：

$$
z=\sum_{j=1}^{n}\phi(k_j)
$$

于是最终可以写成：

$$
o_i \approx \frac{\phi(q_i)^\top S}{\phi(q_i)^\top z + \varepsilon}
$$

这里加一个很小的 `\varepsilon`，是为了避免分母太小导致数值不稳定。

这就是 linear attention 最核心的形状。

它的意义非常明确：

- `S` 负责存“历史 key 和 value 的压缩摘要”
- `z` 负责存“历史 key 的归一化摘要”
- 当前 query `q_i` 只需要和这个摘要交互一次

于是原本每个 token 要和所有历史 token 两两交互，变成了：

1. 先把所有历史信息压成一个固定大小状态；
2. 再用当前 query 去读这个状态。

## 5. 矩阵形式更直观

把所有 token 一次写完，会更好看。

令：

$$
\Phi(Q)\in\mathbb{R}^{n\times r},\quad \Phi(K)\in\mathbb{R}^{n\times r}
$$

其中 `r` 是特征映射后的维度。  
那么可以写成：

$$
O = \frac{\Phi(Q)\left(\Phi(K)^\top V\right)}{\Phi(Q)\left(\Phi(K)^\top \mathbf{1}\right)}
$$

这里：

- `\Phi(K)^\top V \in \mathbb{R}^{r \times d_v}`
- `\Phi(K)^\top \mathbf{1} \in \mathbb{R}^{r}`

` \mathbf{1}` 是全 1 向量，表示对所有 key 做求和。

这个式子非常关键，因为它直接告诉你复杂度怎么降下来：

- `\Phi(K)^\top V` 只需要扫一遍序列
- `\Phi(K)^\top \mathbf{1}` 也只需要扫一遍序列
- 最后 `\Phi(Q)` 再扫一遍序列

所以长度 `n` 上的复杂度从二次变成线性。

## 6. 复杂度到底降在哪里

标准 attention 的主要开销是构造 `n \times n` 的注意力矩阵。

如果忽略常数，复杂度可以粗略看成：

$$
O(n^2 d_k + n^2 d_v)
$$

而 linear attention 先算：

$$
\Phi(K)^\top V
$$

这是 `r \times d_v` 的累积，代价约为：

$$
O(n r d_v)
$$

再算：

$$
\Phi(Q)(\cdot)
$$

代价约为：

$$
O(n r d_v)
$$

所以只要 `r` 是固定的，长度维度就是线性的：

$$
O(n)
$$

严格说，上面只计算 attention mixing 的主要项；生成 $\Phi(Q)$ 和 $\Phi(K)$ 还会带来特征映射本身的成本。若 $r\approx d_k\approx d_v\approx d$，常见写法是 $O(nd^2)$。所谓“线性”是指它关于序列长度 $n$ 线性，并不是其他维度都消失了。

## 7. causal linear attention 为什么还能递推

如果是自回归生成场景，我们通常要求不能看未来 token。  
这时 linear attention 还有一个额外好处：它可以递推更新。

定义到时刻 `t` 为止的状态：

$$
S_t = \sum_{j=1}^{t}\phi(k_j)v_j^\top
$$

$$
z_t = \sum_{j=1}^{t}\phi(k_j)
$$

那么它们的递推式就是：

$$
S_t = S_{t-1} + \phi(k_t)v_t^\top
$$

$$
z_t = z_{t-1} + \phi(k_t)
$$

当前输出则是：

$$
o_t = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t + \varepsilon}
$$

这意味着在推理时，你不需要保存完整的 `KV cache`，只要保存固定大小的 `S_t` 和 `z_t`。

这就是 causal linear attention 在长上下文生成里特别有吸引力的原因：**记忆占用和长度无关，而是和特征维度有关。**

更严格地说，这个“固定大小状态”直接对应流式执行和逐 token 解码。如果训练时为了并行而保存每个位置的前缀状态，中间激活仍可能达到 $O(nrd_v)$；因此不能把推理阶段的常数状态，直接等同于所有训练实现都只占常数内存。

## 8. 一个最小实现长什么样

下面这个伪代码能把上面的公式对应起来：

```python
import torch

def phi(x):
    return torch.nn.functional.elu(x) + 1.0


def linear_attention(q, k, v, eps=1e-6):
    # q: [B, H, N, R]
    # k: [B, H, N, R]
    # v: [B, H, N, D]
    q = phi(q)
    k = phi(k)

    kv = torch.einsum("bhnr,bhnd->bhrd", k, v)
    z = k.sum(dim=-2)

    numerator = torch.einsum("bhnr,bhrd->bhnd", q, kv)
    denominator = torch.einsum("bhnr,bhr->bhn", q, z)

    return numerator / (denominator[..., None] + eps)
```

这段代码的结构非常清楚：

- `kv` 就是 `\sum_j \phi(k_j)v_j^\top`
- `z` 就是 `\sum_j \phi(k_j)`
- 分子和分母都只做一次聚合读出

如果换成 causal 版本，就把 `kv` 和 `z` 改成逐步累积即可。

## 9. 两条不同路线：换一个核，还是近似 softmax

linear attention 的成败，很大程度上取决于 `\phi(\cdot)` 怎么选，但常见方法其实有两种不同目标。

### 直接定义可分解的注意力核

Katharopoulos 等人的 Linear Transformer 使用可分解的相似度，常见特征映射是：

$$
\phi(x)=\operatorname{ELU}(x)+1
$$

它得到的是一种新的 kernel attention，通常不等于标准 softmax attention。正值映射可以让分母更容易保持稳定，但核的选择也直接限制了模型能表达怎样的相似度。

### 用随机特征近似 softmax 核

softmax 分子对应指数点积核：

$$
\exp(q^\top k/\sqrt{d_k})
$$

Performer 使用 FAVOR+ 正交随机特征近似这个核，使它也能写成 $\phi(q)^\top\phi(k)$。它的目标是近似常规 full-rank softmax attention，而不是随意换一种相似度。

所以更准确的总结是：**Linear Attention 是一类依靠可分解特征映射实现线性计算的机制；其中有些方法重新定义注意力核，有些方法专门近似 softmax 核。**

## 10. 与 KV cache 的差异

标准 Transformer 做自回归推理时，需要保存每层所有历史 token 的 key 和 value：

$$
K_{\le t}\in\mathbb{R}^{t\times d_k},\qquad V_{\le t}\in\mathbb{R}^{t\times d_v}
$$

缓存大小随上下文长度 $t$ 线性增长。Linear Attention 保存的则是：

$$
S_t\in\mathbb{R}^{r\times d_v},\qquad z_t\in\mathbb{R}^{r}
$$

只要 $r$ 和 $d_v$ 固定，这部分状态相对于序列长度就是常数级。

但“历史被压缩”既是优点，也是能力瓶颈。标准 Attention 可以针对当前 query 重新访问某个具体历史 token；Linear Attention 只能从聚合状态中读出历史。一旦不同 token 的信息在有限状态中互相干扰，后续 query 就很难把它们重新分开。

## 11. 一个三个 token 的直觉例子

假设某个 query $q$ 对三个历史 token 的核权重分别是 $w_1,w_2,w_3$。普通的归一化加权输出为：

$$
o=\frac{w_1v_1+w_2v_2+w_3v_3}{w_1+w_2+w_3}
$$

如果 $w_j=\phi(q)^\top\phi(k_j)$，分子就能写成：

$$
w_1v_1+w_2v_2+w_3v_3=\phi(q)^\top\left[\phi(k_1)v_1^\top+\phi(k_2)v_2^\top+\phi(k_3)v_3^\top\right]
$$

括号里的内容与当前 query 无关，因此历史 token 可以先把信息写入同一个状态：

$$
S=\sum_j\phi(k_j)v_j^\top
$$

query 到来后只需读取一次 $S$，不再重新逐个扫描 $k_1,k_2,k_3,\ldots,k_n$。这就是“每个 query 与所有 key 交互”变成“query 读取累积状态”的最小例子。

## 12. 它为什么有用，又为什么不完全等价

linear attention 的优点很直接：

- 不需要显式构造 `n \times n` 注意力矩阵
- 长序列时更省显存、更省时间
- causal 场景下可以递推，适合流式生成

但它也有明显局限：

- 历史信息被压缩成固定状态，表达力弱于完整 softmax attention
- 重新定义核的方法不等于 softmax；随机特征方法则只是在近似 softmax
- 如果任务特别依赖“精确地回看某个历史 token”，linear attention 可能不如标准 attention
- 理论复杂度更低不保证短序列上墙钟时间更快，实际表现还取决于特征维度和硬件算子

所以它更像一种取舍：  
你用更低的代价，换取更好的长序列可扩展性。

## 13. 和“先算 `K^T V`”的直觉到底差在哪

这个问题值得单独收一下。

你的直觉其实抓到了“先聚合后读取”的骨架。  
但标准 attention 里，不能直接聚合原始 `K` 和 `V`，因为 softmax 会把每个 query 对历史 token 的偏好重新归一化。

linear attention 做的事情是：

1. 不保留逐对的相似度矩阵；
2. 选择或近似一个能写成 `\phi(q)^\top\phi(k)` 的核；
3. 再利用结合律，把历史项压成状态 `S` 和 `z`。

所以它本质上不是“把 KV 提前算掉”，而是“把 attention 改成可累积的核回归”。

## 14. 一句话总结

如果你只记一句话，那就是：

**linear attention 不是简单换一下矩阵乘法顺序，而是选择或近似一个可分解的注意力核，从而把原本显式的 `n \times n` 交互，改成固定状态的累积与读取。**

它适合长序列、流式生成、低显存推理；  
但它也不是标准 attention 的完全替代，因为它牺牲了一部分精确的全局匹配能力。

## 参考链接

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)
- [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794)

