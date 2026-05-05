# GroupDPO：把 DPO 从单对偏好扩展到组级偏好训练


### **这篇论文提出 GroupDPO：把 DPO 从“一正一负”的单对偏好训练扩展到“多正样本、多负样本”的组级偏好训练，并用一阶梯度等价的省显存 surrogate 实现，使 group-wise preference optimization 在大 group size 下更可行。**

论文关注的问题是：现有 DPO 类偏好优化方法通常只使用每个 prompt 下的一条正回答和一条负回答，但实际偏好数据往往包含多个候选回答。把多候选回答压缩成单个 chosen-rejected pair 会浪费大量监督信号，无法充分利用同一 prompt 下多个回答之间的相对质量信息。

为此，论文提出 **GroupDPO**。它将每个 prompt 的回答组织成一个 group，其中包含正样本集合 $P_g$ 和负样本集合 $N_g$。模型不再只比较一对回答，而是联合比较同一 prompt 下的多条正回答和多条负回答。论文将多种 group-wise objective 统一到同一框架中，包括 Margin、MPO、All-Pairs 和 Softmax 等方法。

GroupDPO 的主要挑战是显存开销。因为组级损失会让同组 responses 的梯度相互耦合，朴素实现需要对整个 group 做联合 forward 和 backward，并保留所有样本的 activation。随着 group size 增大，显存开销快速上升，容易 OOM。

论文的核心技术贡献是提出一种 **memory-efficient surrogate implementation**。其做法是先进行一次 no-gradient forward pass，计算每条 response 的隐式偏好分数 $u_i$，再根据原始 group-level loss 计算每条样本的梯度系数 $c_i$。随后，将组级目标改写为 sample-level surrogate loss：

$$
L_{sur}(\theta)=\sum_i c_i u_i(\theta)
$$

这个 surrogate loss 在当前参数点与原始 group loss 具有相同的一阶梯度，因此可以保持训练方向一致；同时由于反向传播变成逐样本形式，不需要保留整个 group 的联合计算图，从而显著降低 peak memory。

论文还强调 **positive-response NLL** 的重要性。GroupDPO 本身主要优化相对偏好，即让正样本得分高于负样本，但这可能导致正样本 log-prob 下降或训练不稳定。因此作者在正样本上加入 NLL 项，鼓励模型继续保持或提升生成好回答的概率。实验表明，NLL 对性能提升和训练稳定性都非常关键。

实验覆盖 offline 和 online 两种设置。Offline setting 中，作者使用 Dolci-Instruct-DPO prompt，由多个模型生成候选回答，再用 reward model 打分，取 top-k 作为 positive group、bottom-k 作为 negative group。Online setting 中，使用 DAPO-Math-17k，模型采样多个回答后用规则判断数学答案是否正确，并据此划分正负样本。实验模型包括 gemma-3-4b-sft、olmo-3-7b-it-sft、olmo-3.1-32b-it-sft 和 qwen3-4b-base。

主要实验结论有三点。第一，<mark style="background: #FFB8EBA6;">使用多个 responses 的 group-wise training 通常优于普通 single-pair DPO 和 RFT，说明组级偏好监督能提供更丰富的学习信号</mark>。第二，不同 group-wise objective 之间差距相对较小，All-Pairs、Margin、MPO、Softmax 都能取得相近效果；相比纠结具体 objective，是否使用 group-wise 信息更重要。第三，加入 positive NLL 能明显提升稳定性，去掉 NLL 后训练更容易性能下降甚至 collapse。

效率实验显示，朴素 GroupDPO 的显存会随 group size 增大而快速上升，而论文提出的 surrogate 实现可以让 peak activation memory 对 group size 不那么敏感，在只增加一次 no-grad 预计算带来少量延迟的情况下，显著降低显存占用并支持更大的 response group。

整体来看，这篇论文的关键价值在于：它证明了偏好优化中“多候选回答组”的监督信号值得利用，并提供了一个实际可扩展的省显存实现，使 GroupDPO 不只是一个更丰富的目标函数，也成为一种更可落地的大模型对齐训练方法。

