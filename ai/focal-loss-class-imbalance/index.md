# Focal Loss 是什么？为什么它能解决类别不平衡问题？


在目标检测、图像分类、医学影像识别、缺陷检测等任务中，我们经常会遇到一个很典型的问题：**样本分布极不均衡**。

比如在目标检测里，一张图片中真正包含目标的区域可能只有几个，但背景区域却可能有成千上万个。模型训练时，大量“容易分类的背景样本”会主导损失函数，使模型把主要精力放在这些其实已经学会的简单样本上，而不是那些少数但更重要的难样本。

**Focal Loss** 就是为了解决这个问题被提出的。它最早广泛用于目标检测任务，尤其是 RetinaNet 这样的单阶段目标检测器中。它的核心思想很直观：**降低简单样本对总损失的贡献，让模型更关注难分类样本。**

<!--more-->

## 先从 Cross Entropy 说起

在理解 Focal Loss 之前，需要先看普通的交叉熵损失，也就是 Cross Entropy Loss。

对于二分类任务，交叉熵损失可以写成：

$$
CE(p_t) = -\log(p_t)
$$

这里的 $p_t$ 表示模型对真实类别的预测概率。

如果真实类别是正类，那么 $p_t = p$；如果真实类别是负类，那么 $p_t = 1 - p$。也就是说，$p_t$ 可以理解为：**模型对正确答案的置信度**。

当 $p_t$ 越大，说明模型越自信地预测对了，损失越小；当 $p_t$ 越小，说明模型预测错得越离谱，损失越大。

这听起来很合理，但它有一个问题：**即使简单样本的单个损失很小，如果数量特别多，它们仍然会在总损失中占据主导地位。**

举个例子，在目标检测中，背景样本可能远远多于目标样本。虽然每个背景样本都很容易被分对，损失也不大，但成千上万个背景样本加起来，依然会压过少量真正有价值的目标样本。结果就是，模型训练被大量简单负样本带偏，难样本没有得到足够关注。

## Focal Loss 是什么？

Focal Loss 是对 Cross Entropy Loss 的改造。它在交叉熵前面加了一个调节因子：

$$
FL(p_t) = -(1 - p_t)^\gamma \log(p_t)
$$

其中，$\gamma$ 是一个非负超参数，通常称为 focusing parameter，也就是“聚焦参数”。

这个公式的关键在于：

$$
(1 - p_t)^\gamma
$$

这个因子会根据样本的难易程度动态调整损失权重。

当一个样本很容易被模型分对时，$p_t$ 很接近 1，那么 $1 - p_t$ 就很小。此时 $(1 - p_t)^\gamma$ 会把这个样本的损失大幅压低。

当一个样本很难被模型分对时，$p_t$ 比较小，那么 $1 - p_t$ 就比较大。此时这个调节因子不会明显降低损失，模型依然会认真学习这个样本。

因此，Focal Loss 的本质可以概括为一句话：

**让简单样本的损失变得更小，让难样本在训练中占据更高权重。**

## 为什么要用 Focal Loss？

Focal Loss 最重要的价值，是缓解训练过程中的**类别不平衡**和**难易样本不平衡**问题。

类别不平衡很好理解。比如在一个疾病检测任务中，正常样本有 100000 张，异常样本只有 1000 张。如果使用普通交叉熵，模型可能只要倾向于预测“正常”，就能获得很高的整体准确率。但这样的模型对于少数类，也就是我们真正关心的异常样本，可能表现很差。

难易样本不平衡则更隐蔽。即使类别数量看起来没有那么夸张，训练中也可能存在大量容易样本。普通交叉熵会对所有样本一视同仁，只要数量足够多，简单样本依然会主导梯度更新。Focal Loss 则会主动削弱这些已经学会的样本，让模型把更多学习能力用在困难样本上。

这也是为什么 Focal Loss 在目标检测中特别有用。单阶段目标检测器需要在密集候选框上进行分类，大量候选框都是背景。没有 Focal Loss 时，背景样本的数量优势会严重干扰训练；有了 Focal Loss 后，模型可以忽略大量容易背景，更专注于那些容易混淆的目标区域。

## Focal Loss 能解决什么问题？

Focal Loss 主要解决三类问题。

第一类是**正负样本极度不平衡**。在目标检测、异常检测、欺诈检测、医学诊断等任务中，正样本往往很少，负样本极多。普通损失函数容易被多数类主导，导致模型对少数类识别能力不足。Focal Loss 可以降低大量容易负样本的影响。

第二类是**简单样本过多，难样本学习不足**。很多时候，模型很快就能学会大量简单样本，但这些样本仍然持续贡献梯度。Focal Loss 通过调节因子降低简单样本损失，使模型更关注边界样本、低置信度样本和容易混淆的样本。

第三类是**模型训练目标与业务目标不一致**。例如某些任务中，整体准确率并不是最重要的，召回少数类才是关键。医学影像中漏诊一个病灶，可能比误报几个正常区域更严重。Focal Loss 可以帮助模型在训练阶段更加关注这些难识别、少数但重要的样本。

不过需要注意的是，Focal Loss 不是万能的。它主要适用于不平衡明显、难样本重要的任务。如果你的数据分布比较均衡，或者模型本身还没有学到足够基础特征，盲目使用 Focal Loss 可能并不会带来明显提升，甚至会让训练变得更敏感。

## 两个重要参数：alpha 和 gamma

实际使用 Focal Loss 时，最常见的版本会加入一个类别平衡因子 $\alpha$：

$$
FL(p_t) = -\alpha_t(1 - p_t)^\gamma \log(p_t)
$$

这里有两个关键参数：$\alpha$ 和 $\gamma$。

$\gamma$ 用来控制模型对难样本的关注程度。$\gamma = 0$ 时，Focal Loss 就退化成普通交叉熵。$\gamma$ 越大，简单样本的损失被压得越低，模型越关注难样本。常见取值是 $\gamma = 2$。

$\alpha$ 用来平衡类别之间的重要性。例如正样本较少时，可以给正样本更高的 $\alpha$，给负样本较低的权重。这样可以进一步缓解类别不平衡问题。原论文在 RetinaNet 实验中常用 $\alpha = 0.25$、$\gamma = 2$ 作为一组强 baseline，但实际任务中仍然需要根据数据分布和验证集指标调整。

简单来说，$\gamma$ 解决的是“简单样本太多”的问题，$\alpha$ 解决的是“类别数量不均衡”的问题。

## Focal Loss 和 Cross Entropy 的区别

| 对比项 | Cross Entropy | Focal Loss |
|---|---|---|
| 核心思想 | 所有样本按预测误差计算损失 | 降低简单样本权重，关注难样本 |
| 是否处理类别不平衡 | 不直接处理 | 可以通过 $\alpha$ 和 $\gamma$ 缓解 |
| 是否关注难样本 | 间接关注 | 显式关注 |
| 适用场景 | 常规分类任务 | 类别不平衡、难样本多的任务 |
| 参数复杂度 | 简单 | 需要调 $\alpha$ 和 $\gamma$ |

Cross Entropy 更像是一个通用默认选择，适合大多数标准分类任务。Focal Loss 则更像是针对不平衡场景的增强版本。当数据中存在大量容易样本，并且这些样本干扰模型学习时，Focal Loss 往往更合适。

## 如何在项目中应用 Focal Loss？

在实际项目中，不建议一上来就直接使用 Focal Loss。更稳妥的做法是先用 Cross Entropy 或 BCE Loss 建立一个 baseline，然后观察模型是否真的存在不平衡问题。

如果你发现模型在多数类上表现很好，但在少数类上召回率很差，或者模型训练过程中负样本损失占据主导，那么 Focal Loss 就值得尝试。

对于二分类任务，通常可以从下面的设置开始：

$$
\gamma = 2
$$

$$
\alpha = 0.25
$$

如果少数类仍然学得不好，可以适当提高少数类权重。如果模型训练不稳定，或者整体性能下降，可以降低 $\gamma$，例如尝试 $\gamma = 1$。

对于多分类任务，Focal Loss 也可以扩展使用。做法通常是在 softmax 之后取真实类别对应的概率，然后套用类似公式：

$$
FL = -\alpha_y(1 - p_y)^\gamma \log(p_y)
$$

其中 $p_y$ 是模型对真实类别 $y$ 的预测概率，$\alpha_y$ 是该类别对应的权重。

## PyTorch 中的 Focal Loss 示例

下面是一个适用于二分类任务的 PyTorch 实现。输入一般是 logits，也就是模型最后一层没有经过 sigmoid 的输出。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: shape [N] or [N, 1]
        # targets: same shape as logits, values should be 0 or 1
        targets = targets.float()

        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        p_t = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma
        loss = alpha_t * focal_factor * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
```

这个实现里没有直接写 $-\log(p_t)$，而是使用了 `binary_cross_entropy_with_logits`。这是更推荐的写法，因为它在数值上更稳定，尤其是在 logits 很大或很小时，可以减少溢出风险。

使用方式如下：

```python
criterion = BinaryFocalLoss(alpha=0.25, gamma=2.0)
loss = criterion(logits, targets)
loss.backward()
```

如果你做的是多分类任务，可以用下面这种实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: shape [N, C]
        # targets: shape [N], values should be class indices
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)

        focal_factor = (1 - p_t) ** self.gamma
        loss = focal_factor * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
```

使用多分类版本时，可以为每个类别设置不同权重：

```python
alpha = torch.tensor([0.1, 0.3, 0.6])
criterion = MultiClassFocalLoss(alpha=alpha, gamma=2.0)
loss = criterion(logits, targets)
```

这里的 `alpha` 表示不同类别的重要性。通常样本越少的类别，可以设置更高的权重。

## 应用 Focal Loss 时的注意事项

Focal Loss 虽然有效，但使用时有几个常见坑。

首先，不要只看 accuracy。类别不平衡任务中，准确率很容易产生误导。比如一个数据集中 99% 都是负样本，模型全部预测负样本也能有 99% accuracy，但它对正样本完全没有识别能力。更合适的指标包括 precision、recall、F1-score、AUC、PR-AUC，尤其是少数类的 recall 和 F1。

其次，$\gamma$ 不是越大越好。过大的 $\gamma$ 会让模型过度忽略简单样本，导致训练不稳定，甚至让模型基础分类能力下降。一般可以从 1 或 2 开始尝试。

再次，$\alpha$ 不要机械地按类别频率设置。有些人会直接用类别频率的倒数作为权重，但这可能导致少数类权重过大，模型过度预测少数类。更稳妥的做法是根据验证集指标逐步调整。

最后，Focal Loss 可以和其他策略结合使用。例如重采样、数据增强、hard negative mining、类别权重、阈值调整等。但不要一次引入太多策略，否则很难判断到底是哪一个方法带来了效果提升。

## 什么时候应该使用 Focal Loss？

如果你的任务符合下面几种情况，Focal Loss 通常值得尝试：

- 正负样本比例非常悬殊，例如 1:100、1:1000，甚至更极端；
- 模型在多数类上表现很好，但少数类召回率很低；
- 训练过程中大量简单负样本占据主要损失；
- 任务属于目标检测、异常检测、缺陷检测、医学影像识别、欺诈检测等场景；
- 你更关心难样本或少数类，而不是整体准确率。

相反，如果你的数据集比较均衡，或者模型仍然处在欠拟合状态，Focal Loss 未必是第一选择。此时更应该先检查模型结构、学习率、数据质量、特征表达和训练流程。

## 一个直观类比

可以把模型训练想象成老师给学生讲题。

Cross Entropy 的做法是：所有错题和对题都纳入教学，只要题目数量多，就会占用很多课堂时间。于是大量简单题虽然已经会了，但仍然不断出现，占据了教学资源。

Focal Loss 的做法是：学生已经会的简单题，少讲一点；学生总是错的难题，多讲一点。这样训练资源就被重新分配到了更有价值的地方。

这就是 Focal Loss 的“focal”含义：**把学习焦点放到真正困难、真正重要的样本上。**

## 总结

Focal Loss 并不是一个复杂的损失函数，但它解决了深度学习训练中非常实际的问题：当简单样本太多、类别极不平衡时，普通交叉熵容易被大量简单样本主导，导致模型忽视少数类和难样本。

它通过引入 $(1 - p_t)^\gamma$ 这个调节因子，动态降低容易样本的损失贡献；再通过 $\alpha$ 平衡不同类别的重要性。对于目标检测、异常检测、医学影像、缺陷识别等任务，Focal Loss 往往是一个非常值得尝试的选择。

真正使用时，建议从 Cross Entropy 建立 baseline，再引入 Focal Loss，并重点观察少数类 recall、F1、PR-AUC 等指标。只有当你的问题确实存在类别不平衡或难样本学习不足时，Focal Loss 才能发挥它最大的价值。

## 参考链接

- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

