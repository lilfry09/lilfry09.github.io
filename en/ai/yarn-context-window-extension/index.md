# Read YaRN: Long context expansion, the core is not to pull the window hard, but not to break the RoPE


It would be a pity if we only regard `YaRN` as "another trick to pull the context from `4k` to `64k/128k`".

What it really encounters is a more essential geometric problem than "window expansion":

**The model is already accustomed to a position and phase system. Now that you forcefully increase the length, how can you make this system continue to work at a longer range while trying not to lose the positional resolution at close range? **

Once this question is clarified, `YaRN`’s design will appear very natural.

<!--more-->

## 1. Task

Take away all the terminology, and the problem `YaRN` wants to be solved can be formalized into one sentence:

**Given a `RoPE`-based Transformer trained on the length `L`, find a new position mapping so that the model can still distinguish positions stably on the length `sL` and try to maintain the original attention geometry in the short-range interval. **

More specifically, the original position encoding can be viewed as a set of phase systems with frequency `omega_i`.
The original phase is:

`theta_i(p) = omega_i * p`

Now we need to find a new phase function `theta'_i(p)`, which satisfies three conditions:

1. When `p` is within the original training interval `[0, L]`, `theta'_i(p)` and `theta_i(p)` try to be consistent.
2. When `p` is extended to a longer interval `[L, sL]`, the phases should not be seriously aliased, and the model can distinguish different positions.
3. For small relative displacements `delta p`, especially local order relationships, the resolution cannot be significantly squashed.

If we write at the model level, it can be understood that the new attention core `K'(p, q)` must satisfy:

- In the short range, `K'(p, q)` tries to approximate the original `K(p, q)`
- In the long run, `K'(p, q)` still remains distinguishable and generalizable
- Keep training and inference costs as small as possible

So this is not a simple "increase the window number" problem, but a multi-scale geometric continuation problem with constraints.

## 2. Challenge

The most natural idea of ​​​​the traditional method is `Positional Interpolation (PI)`:

- It turns out that the model only handles length `L`
- Now want to handle the length `sL`
- Then press the longer position back to the original coordinate range

This idea is intuitively beautiful, but it has a first-principles flaw:

**It treats all frequency dimensions as the same thing. **

But `RoPE` is not a ruler, but a whole set of rulers with different frequencies:

- The high-frequency dimension is more like a microscope and is responsible for close range and fine-grained locations
- The low-frequency dimension is more like a telescope, responsible for long distances and coarse-grained ranges

If all frequencies are linearly compressed together, three problems arise:

1. The long-range range has been expanded, but the short-range resolution has also been smoothed out.
2. High-frequency components are distorted first, and the sense of local order is most easily damaged.
3. The position transformation is fixed, while the autoregressive generation length grows dynamically, so the two naturally do not match.

Later `NTK-aware` has realized that "one size does not fit all", but it is more like a global correction and has not completely restored the problem to a multi-scale system.

Therefore, the real stuck point of the traditional method is not "not being able to expand the window", but:

**Does not handle gracefully the different responsibilities of different frequencies over short and long range. **

## 3. Insight & Novelty

### 3.1 Inspiration

The inspiration behind `YaRN` essentially comes from two very common-sense intuitions.

First, the intuition of signal processing:

- High frequency is responsible for details
- Low frequencies are responsible for contouring
- When resampling, I am most afraid of erasing high-frequency details first.

Second, the intuition of the generation process:

- Autoregressive reasoning is not a jump to `128k`
- Instead, it grows little by little from short to long.

Put these two common sense together, and they are very close to the core of `YaRN`.

### 3.2 Insight

I think there are three insights that are really valuable in this paper.

#### Insight 1: The long context problem is first of all the frequency distortion problem

This Insight is inspired by the signal processing intuition of "high frequency preserves detail, low frequency preserves range".

It corrects a common misunderstanding:

> The long context does not simply pull the position from `4k` to `64k`, but determines which frequencies should be maintained, which frequencies should be extended, and which frequencies should be transitioned.

This is an Insight about **positional geometry**.

#### Insight 2: Different frequencies should not be subjected to the same stretch

This Insight also comes from multi-scale signal intuition, but it goes one step further.

If high frequency is mainly responsible for local relationships, it should be protected as much as possible;
If the bass is primarily responsible for long range, it is better suited to take on the extension;
The intermediate frequencies make a smooth transition.

This is an insight into the multi-scale division of responsibilities.

#### Insight 3: Length scaling should follow the actual generated length

This Insight is inspired by the intuition that the generation length grows dynamically.

Since the model naturally expands from short to long during inference, the position scaling should not be fixed from beginning to end.
When it is short, it will be distorted prematurely, resulting in a pure loss of local resolution.

This is an insight into the way inference processes are coupled to location systems.

### 3.3 Novelty

The author's Novelty is not mainly at the architecture level, but at the **method level and strategy level**.

No new Transformer was invented, and the main structure of attention was not changed.
The real innovation lies in: a set of more detailed and smoother correction solutions is provided around the multi-scale distortion problem of `RoPE`.

Write strictly according to your format:

[Unified interpolation will flatten local position signals] -> [Inspired by Insight 1 and Insight 2: frequency responsibilities are different and cannot be processed uniformly] -> [Designed `NTK-by-parts` style segmented frequency scaling: high frequencies move as little as possible, low frequencies bear long-range expansion, and intermediate frequencies transition smoothly]

[Fixed scaling factors will allow short sequences to withstand long context distortion in advance] -> [Inspired by Insight 3: The generation length is inherently growing dynamically] -> [The `Dynamic Scaling` / `Dynamic YaRN` idea is designed to update the scaling factor according to the current real sequence length at each forward, rather than fixing a ratio throughout the process]

[After the position coordinate system changes, the numerical distribution of attention logits will also change] -> [Inspired by Insight 1: Geometric changes will eventually be reflected as attention numerical changes] -> [Add attention temperature / length scaling correction to make the softmax behavior under long context more stable]

[Long context adaptation often requires a large number of training tokens and training steps] -> [Inspired by the previous three Insights: What really needs to be repaired is the position system, rather than retraining the entire model] -> [Concentrate the innovation points as much as possible on `RoPE` and its supporting scaling strategy, so that training is cheap and inference has almost no additional overhead]

## 4. Potential Flaw

### 4.1 Limitations of the current situation and whether it can be extended to new situations

The scenario that `YaRN` addresses has clear boundaries:

- The model is `RoPE`-based Transformer
- The main problem is that the position geometry is distorted when extended over long distances
- Hope to extend the context at a very low cost

This is strong, but also very local.

It does not directly answer these more difficult scenarios:

- What if the model is not `RoPE`, but another location system?
- What if it is multi-dimensional position coordinates such as video, audio, and 3D scenes?
- What if not only the length is getting longer, but the conditions, constraints, and modes are also changing?

These can all be extended:

- Anisotropic expansion of multidimensional `RoPE` or Fourier position systems
- Time-space joint scaling in long video modeling
- Joint design of retrieval, memory, compression modules and location systems

### 4.2 It will be particularly difficult if the data has any bad properties

In the current situation, if the data has the following properties, the problem will become significantly more difficult:

- Distant fragments are highly similar but semantically very subtly different.
- Key evidence is buried in local corners of extremely long texts, requiring very detailed positioning.
- The task is not to “see further” but to “do combinatorial reasoning across great distances”
- There are a lot of templated repetitions in the text, which can easily cause the model to mistake positional similarity for semantic similarity.

At this time, even if the position system is not broken, the model may still not be used.
Because the difficulty has shifted from "coordinate distortion" to "information selection, memory allocation, and long-range reasoning."

### 4.3 Which difficulty is most worth digging into a paper?

The most worthy of digging into is the following:

**How ​​to jointly optimize position system correction, long text training signals, and the model's actual ability to utilize ultra-long dependencies. **

This thing has thesis value because it just fills in the boundary of `YaRN`:

- `YaRN` The solution is "Don't let the coordinates break first"
- The next more in-depth paper should address "After the coordinates are fixed, how does the model really learn to use them?"

## 5. Motivation

If we think from first principles, the general idea of ​​`YaRN` can actually be naturally derived from a series of questions:

- The previous method only compressed the position uniformly, so why is it easy to break once it is expanded to a very long position?
- What is bad is "insufficient length" or "positional geometric distortion"?
- If `RoPE` is essentially a set of phase systems at different frequencies, why do all frequencies suffer the same stretch?
- High frequency is already serving local details, so can we use it as little as possible?
- Low frequency is already in the long-range service range, so is it more suitable for window expansion?
- If the frequency in between is responsible for both local and long-range, should it be a smooth transition instead of a one-size-fits-all approach?
- Since autoregressive generation is originally from short to long, why should the scaling factor be fixed throughout?
- If the position coordinates change, the numerical temperature of the attention will also change. Is it possible to adjust the temperature at the same time?
- If what really needs to be modified is the position system, rather than the entire model, can we focus all the changes on the smallest circle?

Following these questions, `YaRN` is almost the most natural and effortless answer.

## Reference link

- [YaRN: Efficient Context Window Extension of Large Language Models - arXiv](https://arxiv.org/abs/2309.00071)
- [YaRN - ar5iv HTML version](https://ar5iv.labs.arxiv.org/html/2309.00071)
- [Extending Context Window of Large Language Models via Positional Interpolation - arXiv](https://arxiv.org/abs/2306.15595)

