# KV Cache 优化地图：从丢 Token 到跨层共享


长上下文推理最先撞到的墙，经常不是 FLOPs，而是 KV Cache。

模型每生成一个新 token，都要让当前 query 访问过去 token 的 key 和 value。为了避免把过去全部重算一遍，推理系统把这些 K、V 留在显存里。代价是缓存随上下文长度、层数和 KV head 数一起增长。

这类工作看起来很多，其实可以放进一个很清楚的坐标系：

> KV Cache 优化主要是在 Token、Layer、Head 三个维度上减少需要保存的状态。

<!--more-->

## KV Cache 为什么会这么大

标准 Transformer 的每一层都有自己的 $W_K$ 和 $W_V$。第 $\ell$ 层对第 $i$ 个 token 产生：

$$
K_i^{(\ell)} = h_i^{(\ell)}W_K^{(\ell)}, \qquad
V_i^{(\ell)} = h_i^{(\ell)}W_V^{(\ell)}
$$

因此缓存并不是“最后一层的一份 K、V”，而是每个需要独立 KV 的层都保存一份历史状态。忽略实现细节后，缓存字节数近似为：

$$
2BLNH_{kv}d_h \times \text{bytes per element}
$$

其中 $B$ 是 batch size，$L$ 是层数，$N$ 是序列长度，$H_{kv}$ 是 KV head 数，$d_h$ 是 head dimension，前面的 2 对应 K 和 V。

于是压缩路径自然分成三类：

| 维度 | 要减少什么 | 代表方法 |
|---|---|---|
| Token | 每层保留的历史位置数 $N$ | StreamingLLM、H2O、SnapKV、InfLLM |
| Layer | 独立产生和保存 KV 的层数 $L$ | CLA、LCKV、YOCO |
| Head | 每层独立 KV head 数 $H_{kv}$ | MQA、GQA |

这三个维度可以组合，但代价不同：运行时方法通常能直接用于已有模型，架构方法往往需要重新训练。

## Token 维度：哪些历史位置值得留

### StreamingLLM：保留 sink，再保留最近窗口

普通滑动窗口只留下最近 token，序列一旦超过窗口，模型质量可能突然崩掉。StreamingLLM 发现，模型会把一部分注意力稳定地分配给最开头的少数 token，即使这些 token 本身没有特别强的语义，这些位置也像承接多余注意力质量的“水槽”。

它因此保留两部分缓存：

- 开头少量 attention sink token；
- 最近的一段滑动窗口。

中间位置被驱逐，缓存大小不再随总生成长度增长。论文报告，这种无需微调的做法能让预训练模型稳定处理数百万 token 的持续流，并相对窗口重计算基线取得最高 22.2 倍加速。它适合持续对话和流式文本，但被丢掉的中间细节无法再找回。

### H2O：保留真正贡献大的 Heavy Hitters

StreamingLLM 固定保护开头，H2O 进一步问：除了最近位置，能不能根据历史注意力动态判断哪些 token 最重要？

H2O 累积 token 获得的注意力贡献，保留近期 token 和少量 heavy hitters，再驱逐其余位置。论文把这个过程写成动态子模优化问题，并在 OPT、LLaMA、GPT-NeoX 上验证。

它比固定窗口更适合重要信息分散在上下文中的情况，但评分依赖已经发生过的注意力。如果某个早期信息暂时不受关注、之后才突然重要，驱逐决定仍可能做错。

### SnapKV：在生成前预测 prompt 中的重要位置

SnapKV 针对的是“长 prompt、后续生成相对较短”的场景。它观察到，不同 attention head 在生成期间关注的 prompt 位置相对稳定，而且可以用 prompt 末端的一小段 observation window 提前估计。

方法先让观察窗口对完整 prompt 投票，再按 head 选择重要位置并聚类，只保留这些位置的 KV。论文报告，在 16K 输入上获得 3.6 倍生成加速和 8.2 倍内存效率提升，并在单张 A100 80GB 上处理约 380K token。

它压缩的是已经完成 prefill 的 prompt cache。若任务在生成过程中不断改变关注目标，最初的选择可能不再适合后面的 token。

### InfLLM：远处信息不删除，转入可检索记忆

StreamingLLM、H2O 和 SnapKV 都在决定“留下谁”。InfLLM 走的是另一条路：只在活跃窗口中保留少量上下文，把远距离信息组织成 memory units，需要时再检索回来。

这让运行成本主要跟活跃窗口有关，同时保留访问远处信息的可能。论文报告，无需继续训练即可把原始短上下文模型扩展到最长 1,024K token 的长距离依赖测试。

但外部记忆并没有让信息免费存在。它把全量 attention 的成本换成了记忆组织、索引和检索成本，最终效果取决于相关片段能否被正确找回。

## Layer 维度：每一层都必须存自己的 KV 吗

### CLA：相邻层共享

MQA 和 GQA 在一层内部减少 KV heads；Cross-Layer Attention 把共享扩展到深度维度，让相邻若干层复用同一组 key 和 value。

论文在 1B 和 3B 模型上从头训练，报告 CLA 能在 MQA 基础上进一步把 KV Cache 缩小约 2 倍，同时保持接近的准确率。它的优点是结构变化温和，代价是跨层共享会减少每层独立表示历史信息的自由度。

### LCKV：只让少数层形成缓存

Layer-Condensed KV Cache 更激进：只有少量层独立计算和缓存 K、V，其他层使用压缩后的跨层缓存结构。缓存不再与全部层数线性增长。

论文报告了显著的内存下降和最高 26 倍吞吐提升，但这类数字强烈依赖模型规模、batch、上下文和基线。更重要的是，LCKV 改变了层间信息流，需要专门训练，不能像运行时驱逐策略一样直接套在任意已有权重上。

### YOCO：把模型拆成“构造记忆”和“读取记忆”两段

YOCO 不是简单让相邻层共享，而是重新设计 decoder：

- self-decoder 负责处理序列并形成一份全局 KV Cache；
- cross-decoder 的多层 query 都读取这份共享缓存。

模型对外仍按自回归方式生成，但历史 K、V 只需缓存一次。这个分工还允许 prefill 提前结束部分计算，从而同时降低缓存占用和首 token 前的延迟。论文把 YOCO 扩展到 1M 上下文，并报告在长上下文检索与系统性能上保持竞争力。

它获得更大收益的原因也正是它改得更深：需要按 YOCO 架构训练模型，而不是只替换推理引擎里的缓存策略。

## Head 维度：MQA 与 GQA 是另外一条轴

标准 Multi-Head Attention 往往为每个 query head 保存独立 K、V。Multi-Query Attention 让全部 query heads 共享一组 KV；Grouped-Query Attention 则在两者之间，让一组 query heads 共享 KV。

这一轴和 token、layer 压缩相对独立。理论上可以组合成：

- GQA 减少每层 KV heads；
- CLA 减少独立 KV layers；
- H2O 或 SnapKV 减少每层保留的 token。

但压缩收益不能简单相乘。每种方法都在减少表示容量，叠加后必须重新验证长距离召回、困惑度、生成质量和真实吞吐。

## 怎么选择方法

如果不能重新训练模型，优先看 token 维度：

- 持续流式生成：StreamingLLM；
- 重要位置动态变化：H2O；
- 长 prompt 后生成：SnapKV；
- 超长文档且需要找回远处内容：InfLLM。

如果可以从架构和训练阶段介入，再考虑 layer/head 维度：

- 温和减少缓存：GQA、CLA；
- 追求更高跨层共享：LCKV；
- 从模型结构上同时处理共享缓存和 prefill：YOCO。

真正部署时还要同时测四类指标：峰值显存、首 token 延迟、逐 token 吞吐和任务质量。只报告“缓存缩小多少”不足以判断系统是否更快，因为索引、压缩、数据搬运和不规则 attention 都可能吃掉收益。

## 结语

KV Cache 优化最有用的思维模型不是记住一串论文名，而是先问三个问题：

1. 它减少了多少 token 位置？
2. 它减少了多少独立 KV 层？
3. 它减少了多少 KV heads？

接着再问：被删掉或共享的信息，用什么机制补回来？

所有方法都在用冗余换显存、速度和长度。差别只在于，它认为冗余存在于 token、layer 还是 head，以及愿意为这次压缩承担多少质量风险和训练成本。

## 参考链接

- [StreamingLLM](https://arxiv.org/abs/2309.17453)
- [H2O: Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048)
- [SnapKV](https://arxiv.org/abs/2404.14469)
- [InfLLM](https://arxiv.org/abs/2402.04617)
- [Cross-Layer Attention](https://arxiv.org/abs/2405.12981)
- [Layer-Condensed KV Cache](https://arxiv.org/abs/2405.10637)
- [YOCO](https://arxiv.org/abs/2405.05254)
- [Multi-Query Attention](https://arxiv.org/abs/1911.02150)
- [Grouped-Query Attention](https://arxiv.org/abs/2305.13245)

