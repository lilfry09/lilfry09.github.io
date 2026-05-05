# High frequency interview: Bradley-Terry vs Plackett-Luce, what is the difference between reward modeling?


To sum up their relationship in the most straightforward sentence: **The Bradley-Terry (BT) model is a special case of the Plackett-Luce (PL) model when the candidate set size is $N=2$. ** They are essentially designed to map human subjective "preference" or "ranking" into a continuous "reward score (Reward)".

For those who are used to writing PyTorch, you can understand them as **"two-class Sigmoid"** and **"multi-class sequence Softmax"** respectively.

The following is a detailed comparison of their advantages, disadvantages and scope of application:

### 1. Bradley-Terry (BT) model: the cornerstone of pairwise comparisons

In the standard alignment process, the BT model is what we most often come into contact with. It assumes that when comparing two answers $y_w$ (winner) and $y_l$ (loser), the probability of a human choosing $y_w$ depends only on the difference in their potential reward scores $r$:

$$P(y_w \succ y_l) = \frac{\exp(r(y_w))}{\exp(r(y_w)) + \exp(r(y_l))} = \sigma(r(y_w) - r(y_l))$$

**Pros:**
* **The cost of data annotation is extremely low:** Letting human annotators (or GPT-4 as a referee) judge "Which A or B is better" is the task with the least cognitive load, so the open source community has accumulated a large amount of Pairwise preference data (such as the data set used in your project).
* **Elegant mathematical properties and easy to implement in engineering:** Because it is essentially a Log-Sigmoid loss function. The DPO algorithm can be derived so simply because of the mathematical form of the BT model, which perfectly offsets the partition function and avoids online sampling.
* **High computational efficiency:** Only one pair of inputs are required when calculating Loss, and the video memory usage is relatively small.

**Disadvantages (Cons):**
* **Low information utilization:** It can only handle "one-on-one" confrontation and cannot directly capture global sorting information.
* **Transitivity Violation:** In reality, human preferences are often not strictly transitive (for example, A wins B, B wins C, but humans may feel that C wins A). The BT model forcibly compresses preferences onto the one-dimensional real axis $r$, forcing transitivity, which may lead to inevitable deviations in the model when fitting complex preferences.

**Scope of application:**
* Standard large model RLHF (training baseline Reward Model).
* Standard DPO and its derivative algorithms (such as IPO, KTO).
* Ranking system for competitive games (the Elo rating system is actually a variant of the BT model).

---

### 2. Plackett-Luce (PL) model: an advancement of global sorting

The PL model was born to solve **Listwise Ranking**. Suppose now $K$ responses are generated for the same prompt, and let humans rank them from best to worst: $y_1 \succ y_2 \succ \dots \succ y_K$.
The logic of the PL model is: first select the 1st place from $K$, then select the 2nd place from the remaining $K-1$, and so on. The joint probability of this sorting result is:

$$P(y_1 \succ y_2 \succ \dots \succ y_K) = \prod_{j=1}^K \frac{\exp(r(y_j))}{\sum_{m=j}^K \exp(r(y_m))}$$

**Pros:**
* **Data utilization efficiency is extremely high:** One Listwise sorting annotation contains much more information than splitting it into multiple independent Pairwise comparisons. The model can "see" good, medium, and poor distributions at the same time, and the direction of gradient update is more accurate, which can significantly alleviate the local overfitting or scoring scale drift problems we encountered when training the Reward Model.
* **More in line with actual application scenarios:** In multi-agent generation, search engines or recommendation systems, what we often need is to sort a candidate list, not just to judge which of the two is better.

**Disadvantages (Cons):**
* **Data acquisition is extremely difficult:** For humans to sort 5 long texts from 1 to 5, the cognitive load is extremely high, resulting in extremely poor annotation consistency, and the data is extremely expensive and scarce.
* **The computing and graphics memory overhead increases dramatically:** For each Batch, you need to feed $K$ long texts into Transformer at the same time to calculate Logits. On large models that are often 7B or 14B, it is very easy to cause OOM (video memory overflow).
* **The elegant derivation of DPO cannot be directly reused:** After substituting the PL model into the RL objective function, the partition function cannot be eliminated as easily as the BT model, which makes the direct alignment algorithm based on PL much more complex in both mathematical derivation and engineering implementation.

**Scope of application:**
* **Listwise Alignment Algorithm:** Such as PRO (Preference Ranking Optimization) or LiPO, these are the cutting-edge directions that the academic community is currently trying to surpass DPO.
* Information Retrieval (IR) and Recommender Systems: Learn how to rank search results (e.g. ListMLE loss function).

## Reference links

- [Bradley & Terry (1952): Rank Analysis of Incomplete Block Designs: The Method of Paired Comparisons](https://academic.oup.com/biomet/article/39/3-4/324/326091)
- [Plackett (1975): The Analysis of Permutations](https://academic.oup.com/jrsssc/article/24/2/193/6953554)
- [Luce (1959): Individual Choice Behavior: A Theoretical Analysis](https://openlibrary.org/books/OL4735832M/Individual_choice_behavior)
- [DPO: Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [LiPO: Listwise Preference Optimization through Learning-to-Rank](https://arxiv.org/abs/2402.01878)
- [Listwise Approach to Learning to Rank: Theory and Algorithm](https://icml.cc/Conferences/2008/papers/167.pdf)

