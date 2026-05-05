# 面试高频：Bradley-Terry vs Plackett-Luce，奖励建模到底差在哪


用一句最直白的话来总结它们的关系：**Bradley-Terry (BT) 模型是 Plackett-Luce (PL) 模型在候选集大小 $N=2$ 时的特例。** 它们本质上都是为了把人类主观的“偏好”或“排名”，映射为一个连续的“奖励分数（Reward）”。

对于习惯了写 PyTorch 的人来说，你可以把它们分别理解为 **“二分类的 Sigmoid”** 和 **“多分类的序列 Softmax”**。

以下是它们的详细优劣与适用范围对比：

### 1. Bradley-Terry (BT) 模型：成对比较的基石

在标准的对齐流程中，我们最常接触的就是 BT 模型。它假设在比较两个回答 $y_w$ (winner) 和 $y_l$ (loser) 时，人类选择 $y_w$ 的概率只取决于它们的潜在奖励得分 $r$ 之差：

$$P(y_w \succ y_l) = \frac{\exp(r(y_w))}{\exp(r(y_w)) + \exp(r(y_l))} = \sigma(r(y_w) - r(y_l))$$

**优势 (Pros)：**
* **数据标注成本极低：** 让人类标注员（或作为裁判的 GPT-4）判断“A 和 B 哪个更好”是认知负担最小的任务，因此开源界积累了海量的 Pairwise 偏好数据（如你项目中用到的数据集）。
* **数学性质优雅且易于工程实现：** 因为它本质上就是一个 Log-Sigmoid 损失函数。DPO 算法能够如此简洁地推导出来，全靠 BT 模型的这个数学形式，它完美抵消了配分函数（Partition Function），避免了在线采样。
* **计算效率高：** 在算 Loss 时只需要一对输入，显存占用相对较小。

**劣势 (Cons)：**
* **信息利用率低：** 只能处理“一对一”的对抗，无法直接捕捉全局的排序信息。
* **传递性缺陷（Transitivity Violation）：** 现实中人类的偏好往往不具备严格传递性（比如 A 赢 B，B 赢 C，但人类可能觉得 C 赢 A）。BT 模型强行把偏好压缩到一维实数轴 $r$ 上，强制了传递性，这可能会导致模型在拟合复杂偏好时产生不可避免的偏差。

**适用范围：**
* 标准的大模型 RLHF（训练基线 Reward Model）。
* 标准的 DPO 及其衍生算法（如 IPO, KTO）。
* 竞技类游戏的排位系统（Elo 等级分系统其实就是 BT 模型的一种变体）。

---

### 2. Plackett-Luce (PL) 模型：全局排序的进阶

PL 模型是为了解决**列表级排名（Listwise Ranking）**而生的。假设现在给同一个 Prompt 生成了 $K$ 个回答，让人类把它们从最好到最坏排个序：$y_1 \succ y_2 \succ \dots \succ y_K$。
PL 模型的逻辑是：先从 $K$ 个里选出第 1 名，再从剩下的 $K-1$ 个里选第 2 名，以此类推。这个排序结果的联合概率是：

$$P(y_1 \succ y_2 \succ \dots \succ y_K) = \prod_{j=1}^K \frac{\exp(r(y_j))}{\sum_{m=j}^K \exp(r(y_m))}$$

**优势 (Pros)：**
* **数据利用效率极高：** 一次 Listwise 的排序标注，包含的信息量远大于将其拆分成多个独立的 Pairwise 对比。模型能同时“看到”好、中、差的分布，梯度更新的方向更加准确，能显著缓解我们在训练 Reward Model 时遇到的局部过拟合或打分尺度漂移问题。
* **更符合实际应用场景：** 在多智能体生成、搜索引擎或推荐系统中，我们往往需要的是对一个候选列表进行排序，而不仅仅是判断两个谁好。

**劣势 (Cons)：**
* **数据获取极度困难：** 让人类对 5 篇长文本进行 1 到 5 的排序，认知负荷极大，导致标注一致性极差，数据极其昂贵且稀缺。
* **计算与显存开销暴增：** 每一个 Batch 你需要同时把 $K$ 个长文本喂进 Transformer 算 Logits。在动辄 7B、14B 的大模型上，非常容易引发 OOM（显存溢出）。
* **无法直接复用 DPO 的优雅推导：** 把 PL 模型代入 RL 目标函数后，无法像 BT 模型那样轻易地消去配分函数，这导致基于 PL 的直接对齐算法在数学推导和工程实现上都复杂得多。

**适用范围：**
* **Listwise Alignment 算法：** 比如 PRO (Preference Ranking Optimization) 或 LiPO，这些是目前学术界试图超越 DPO 的前沿方向。
* 信息检索（IR）与推荐系统：学习如何对搜索结果进行排序（如 ListMLE 损失函数）。

## 参考链接

- [Bradley & Terry (1952): Rank Analysis of Incomplete Block Designs: The Method of Paired Comparisons](https://academic.oup.com/biomet/article/39/3-4/324/326091)
- [Plackett (1975): The Analysis of Permutations](https://academic.oup.com/jrsssc/article/24/2/193/6953554)
- [Luce (1959): Individual Choice Behavior: A Theoretical Analysis](https://openlibrary.org/books/OL4735832M/Individual_choice_behavior)
- [DPO: Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [LiPO: Listwise Preference Optimization through Learning-to-Rank](https://arxiv.org/abs/2402.01878)
- [Listwise Approach to Learning to Rank: Theory and Algorithm](https://icml.cc/Conferences/2008/papers/167.pdf)

