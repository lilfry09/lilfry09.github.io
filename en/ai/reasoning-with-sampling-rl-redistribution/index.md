# Read "Reasoning with Sampling": RL does not make the model smarter, it just redistributes reasoning capabilities


The really dangerous thing about this paper is not that it proposes a new sampler, but that it asks a deeper interpretive question:

**The progress of reasoning model, is it that the model has learned new capabilities, or have we finally learned how to sample from old capabilities? **

If this question were asked correctly, much of the narrative about `RL for reasoning` would have to be rewritten.

<!--more-->

## 1. Task

Get rid of all appearances, and the real problem this article wants to solve is:

**Given a trained base model `p(x)`, without changing the weights, can we more stably extract high-quality complete inference trajectories through more inference-time compute? **

If formalized a bit, what it does is sample from a new target distribution:

`q_alpha(x) = p(x)^alpha / Z_alpha`

in:

- `x` is the complete output sequence, not a single token
- `alpha > 1` expressed the hope to further amplify the entire trajectory of the original high probability
- `Z_alpha` is the normalization constant

From first principles, this is not a problem of "learning a new model", but a sampling optimization problem with computational budget constraints:

1. The model weights are fixed and parameters are not allowed to be changed.
2. What I want to optimize is the single-shot success rate, not the best pass@k.
3. Allows for more test-time compute.
4. It is hoped that the inherent diversity reserves of the model will not be damaged as much as possible.

In other words, the task of this paper is not "how to continue training", but:

**How ​​to redesign the sampling distribution during inference. **

## 2. Challenge

There are roughly three traditional methods.

The first path is `RL post-training`.

Its logic is very natural:

- A single answer is not strong enough
- Use rewards to adjust weights
- Make the model more likely to go to the right path the first time

The problem is that this path defaults to a very strong premise:

**single-shot becomes stronger, which means the model has learned new abilities. **

This paper suspects that may not be the case.
If the base model can find the correct solution after multiple attempts under `pass@k`, it means that the correct path is already in the distribution.
RL may just refocus the probabilities of these existing paths rather than create new paths.

The second path is low temperature sampling.

The problem is also very fundamental:

- It will indeed make the output more stable
- But it becomes locally greedy token by token
- It optimizes the local next-token distribution, not the entire reasoning trajectory

So it's not "big picture" enough.

If you think about the third way one step further, it is:

- Since the goal is to rearrange the distribution of the entire sequence
- Wouldn't it be better to just do MCMC on the complete sequence?

But this will run into the old problem of high-dimensional sequence space:

- sequence too long
- The state space is too large
- mixing is very slow
- Initialization can easily fall into bad territory

So the real challenge of traditional methods is threefold:

1. The RL is too heavy and the source of the boost is not explained cleanly.
2. The low temperatures are too localized and are not optimizing the complete trajectory.
3. Sampling entire sentences directly is too expensive and too slow.

## 3. Insight & Novelty

### 3.1 What is the author’s inspiration?

I think there are two main inspirations behind this paper.

The first intuition comes from statistical physics/energy models:

- If you want the system to be more high-quality
- It is not necessary to change the system itself
- You can also change the sampling distribution

The second comes from an intuitive observation of the `pass@k` phenomenon:

- If you try a model a few times, you can get it right.
- That means "being able to do it" is already in the model.
- The question may not be "whether you have the ability", but "can you draw that path the first time?"

Once these two intuitions are combined, the entire paper has basically taken shape.

### 3.2 What exactly is the author’s Insight?

I think this paper has at least four layers of insights.

#### Insight 1: `pass@k` is more like capacity reserve, `single-shot` is more like extraction efficiency

This Insight is inspired by the second Inspiration, which is a re-understanding of the `pass@k` phenomenon.

It corrects a common bias:

> The single output we usually see is not necessarily the upper limit of the model's true capability. It is more like the result of the internal capability distribution of the model being mined.

This is an insight into the way capabilities are represented.

#### Insight 2: The benefits of RL may be "redistribution", not necessarily "gain"

This Insight is also inspired by the `pass@k` phenomenon.

If the correct path already exists, then RL is probably doing something simpler:

- The correct probability mass originally dispersed in multiple samples
- Focus on the first answer

This is an insight into **sources of RL improvement**.

#### Insight 3: What should really be optimized is the entire trajectory, not each token step

This insight is closer to inspiration from statistical physics and energy-based models.

If the quality of reasoning belongs to the "quality of the complete solution", then the sampling distribution should also be defined on the complete sequence, rather than on the local tokens of each step.

This is an insight into how the target distribution is defined.

#### Insight 4: The autoregressive structure itself is a bridge for high-dimensional sampling

This insight comes from a common-sense understanding of the LLM generation process itself.

Since the whole MCMC sentence is too difficult, why not use the prefix expansion structure inherent in LLM, first stand firm on low-dimensional short prefixes, and then gradually move towards high-dimensional long prefixes?

This is an insight into how to make high-dimensional sampling computable.

### 3.3 Novelty

The Novelty of this paper is not in the new architecture, but mainly in the **method level and strategy level**.

What it does not do:

- Train a larger model
- Design a more complex reasoning head
- Introduce a heavier RL loop

What it does is:

- Redefine the distribution that should be taken when inferring
- Then design an approximate sampling process that is as computable as possible

Write strictly according to your format:

[RL improves single-shot, but the cost may be diversity collapse, and the source of improvement is not clearly explained] -> [Inspired by Insight 1 and Insight 2: the capabilities may already be there, but the distribution is refocused] -> [The power distribution is designed, that is, directly considering sampling from `q_alpha(x) = p(x)^alpha / Z_alpha`, and changing the goal from "learning new parameters" to "acquiring new distributions"]

[Cryogenic sampling only becomes greedy on local tokens and cannot represent the quality of the entire reasoning trajectory] -> [Inspired by Insight 3: What really should be optimized is the complete trajectory] -> [Define the sampling target on the entire sequence instead of the next-token distribution, thereby turning the reasoning problem into a complete path distribution rearrangement problem]

[The target distribution `p(x)^alpha` cannot be sampled directly] -> [Inspired by Insight 3: Since only relative probability is needed, it can be approximated by a universal sampling method] -> [A power sampling framework based on `Metropolis-Hastings` is designed, a proposal model is used to generate candidates, and then acceptance is decided based on the target distribution ratio]

[MCMC on the entire sequence is too slow to mix in high-dimensional space] -> [Inspired by Insight 4: the autoregressive prefix itself can act as a bridge from low dimension to high dimension] -> [Designed blockwise's intermediate distribution and block-by-block expansion strategy, first adopting short prefixes, and then gradually expanding longer prefixes, and using warm start to avoid direct explosion of whole sentence sampling]

## 4. Potential Flaw

### 4.1 Limitations of the current situation and whether it can be extended to new situations

The premise of this paper is actually quite strong:

- The base model itself already contains a large number of potential correct trajectories
- Trajectories with higher likelihood tend to be closer to high-quality inference
- It is cost-effective to use more test-time compute to search

So its most suitable situation is:

- base model has a good foundation
- The correct solution is already in the distribution
- It’s just that the single-shot is not accurate.

But if you switch to a new, more complex situation, such as:

- multimodal reasoning
- External tool call
- strongly constrained programming
- multi-stage decision making

Then `p(x)^alpha` alone may not be enough.
At this time, you can extend it further:

- Introduce verifiers or external rewards
- Introduce learned proposal
- Adaptive resampling only for key local decisions

### 4.2 It will be particularly difficult if the data has any bad properties

If the data distribution has the following bad properties, the paper's method will be obviously difficult:

- The correct answer itself is not high likelihood, but low likelihood but high correctness
- The model is naturally more confident in misinterpretations that “look smooth”
- Tasks need to rely on external factual verification rather than language fluency
- Critical errors only occur on a few pivot tokens, and resampling the entire suffix is ​​wasteful.

In this case, power sampling will have a very typical frustration:

**It will climb more and more seriously on the wrong but high-probability mountain. **

### 4.3 Which difficulty is most worth digging into a paper?

The one most worthy of digging into, I think, is:

**How ​​to change the sampling target to one closer to the real inference quality when "likelihood is not a good proxy for correctness". **

This line is particularly suitable for continuing to write a paper, because it is stuck on the core and most fragile assumption of the paper.

More specifically, one can go ahead and do:

- verifier-guided power sampling
- adaptive local resampling
- learned proposal + verifier jointly accelerates mixing

## 5. Motivation

If we push the general idea of ​​this article based on first principles, I think the most natural question chain is probably this:

- The previous method mainly relied on RL to improve single-shot. So, is RL creating new capabilities, or is it just rearranging existing capabilities?
- If a base model can answer many questions correctly in `pass@k`, does that mean that the correct trajectory is already in the model distribution?
- If the correct trajectory is already there, then why do we have to change the weights? Why not just change the sampling?
- But if we just lower the temperature, wouldn't we still become greedy locally at each token step?
- Since the quality of reasoning belongs to the entire trajectory, can we directly rearrange the joint probabilities of the entire sequence?
- If you want to prefer high-quality trajectories, wouldn’t the most natural approach be to sharpen `p(x)` into `p(x)^alpha`?
- But this new distribution cannot be sampled directly, so why not consider MCMC?
- But why is the entire MCMC sentence unacceptably slow? Is it because the sequence space is too high-dimensional?
- Since LLM is inherently autoregressive, why not first adopt a reliable state on short prefixes, and then gradually advance to long prefixes?
- If only one block is processed in each step, and a warm start is performed in the previous step, does high-dimensional sampling become a more natural step-by-step correction process?

Following this series of questions, the core idea of ​​this paper appeared almost naturally.

## Reference link

- [Reasoning with Sampling: Your Base Model is Smarter Than You Think - arXiv](https://arxiv.org/abs/2510.14901)
- [Reasoning with Sampling - ar5iv HTML version](https://ar5iv.org/html/2510.14901v1)

