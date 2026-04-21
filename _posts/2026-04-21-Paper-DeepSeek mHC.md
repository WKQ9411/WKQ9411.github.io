---
layout: post
title: 【论文解读】DeepSeek mHC
date: 2026-04-21 16:00:00 +08:00
summary: 本文提出了流形约束超连接（Manifold-Constrained Hyper-Connections, mHC），它将 HC 的残差连接空间投影到特定的流形上，以恢复恒等映射特性。实验表明，mHC 在大规模训练中是有效的，在性能提升的同时具有更优的可扩展性。
categories: Paper
excerpt_separator: <!--more-->
---

{% raw %}
# 目录
{:.no_toc}

* TOC
{:toc}

---

# 概览

以超连接（Hyper-Connections, HC）为代表的研究，通过扩大残差流宽度和多样化连接模式，扩展了经典的残差连接范式。尽管它带来了显著的性能提升，但是也从根本上损害了残差连接固有的恒等映射特性，导致训练不稳定，影响了可扩展性。本文提出了[流形约束超连接（Manifold-Constrained Hyper-Connections, mHC）](http://arxiv.org/abs/2512.24880)，它将 HC 的残差连接空间投影到特定的流形上，以恢复恒等映射特性。实验表明，mHC 在大规模训练中是有效的，在性能提升的同时具有更优的可扩展性。

> 本文仅看 mHC 结构部分，略去原文中关于工程优化的部分。

# 一、介绍

经典的残差网络如下图（a）所示，它可以表示为：
$$
\begin{equation}
\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathbf{W}_l) \label{eq:s_res}
\end{equation}
$$

<div align="center"><img src="https://wkqpicture.oss-cn-beijing.aliyuncs.com/img/20260111170036538.png" width="80%" alt="image-20260111170036421"></div>

其中 $\mathbf{x}_l$ 和 $\mathbf{x}_{l+1}$ 分别表示第 $l$ 层的 $C$ 维输入和输出，$\mathcal{F}$ 表示残差函数。残差连接的恒等映射特性使大规模训练过程能够保持稳定和高效。通过将残差连接递归地扩展到多层，上式可以写为：
$$
\begin{equation}
\mathbf{x}_L = \mathbf{x}_l + \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathbf{W}_i) \label{eq:res}
\end{equation}
$$
其中， $L$ 和 $l$ 分别对应深层和浅层的网络层。恒等映射指的是 $\mathbf{x}_l$ 从较浅层直接传递到更深层次而无需任何修改的特性。

最近，以超连接（Hyper-Connections, HC）为代表的研究，对残差连接进行了扩展。HC 的架构如图（b）所示，它通过拓宽残差流的宽度提高连接复杂度。它可以表示为：

$$
\begin{equation}
\mathbf{x}_{l+1} = \mathcal{H}_l^{\text{res}} \mathbf{x}_l + \mathcal{H}_l^{\text{post} \top} \mathcal{F}(\mathcal{H}_l^{\text{pre}} \mathbf{x}_l, \mathcal{W}_l) \label{eq:hc}
\end{equation}
$$
与经典残差不同，它的特征维度从 $C$ 扩展到 $n \times C$，其中 $n$ 是扩展率（expansion rate）。$\mathcal{H}_l^{\text{res}} \in \mathbb{R}^{n \times n}$ 是一个可学习映射，用于混合残差流中的特征。$\mathcal{H}_l^{\text{pre}} \in \mathbb{R}^{1 \times n}$ 负责将 $nC$ 维的流聚合为 $C$ 维层输入，相反，$\mathcal{H}_l^{\text{post}} \in \mathbb{R}^{1 \times n}$ 负责将层输出映射回流中。

然而，随着训练规模的扩大，HC 带来了潜在的不稳定风险。当扩展到多层时，HC 不受约束的特性会损害恒等映射属性。理想的恒等映射充当着一种守恒机制，它确保在前向和反向传播过程中，各流的平均信号强度保持不变。HC 扩展到多层后，有：
$$
\begin{equation}
\mathbf{x}_L = \left(\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\text{res}}\right) \mathbf{x}_l + \sum_{i=l}^{L-1} \left(\prod_{j=1}^{L-1-i} \mathcal{H}_{L-j}^{\text{res}}\right) \mathcal{H}_i^{\text{post} \top} \mathcal{F}(\mathcal{H}_i^{\text{pre}} \mathbf{x}_i, \mathcal{W}_i)
\end{equation}
$$
与公式 $\eqref{eq:res}$ 不同的是，HC 中的复合映射 $\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\text{res}}$ 无法保留特征的全局均值，这中差异会导致信号不受限制的放大或衰减，从而导致不稳定。另一个需要考虑的因素是，尽管 HC 在 FLOPs 方面保持了计算效率，但原始设计中并未解决加宽的残差流在内存访问成本方面的硬件效率问题。这些因素共同限制了 HC 的实际可扩展性，并阻碍了其在大规模训练中的应用。

为了应对这些挑战，本文提出了流形约束超连接（Manifold-Constrained Hyper-Connections, mHC），如图（c）所示，它将 HC 的残差连接空间投影到特定的流形上，以恢复恒等映射特性，同时配合细致的基础设施优化以确保效率。具体而言，mHC 采用 Sinkhorn-Knopp 算法，将 $\mathcal{H}_l^{\text{res}}$ 投影到 Birkhoff 多面体上。该操作有效地将残差连接矩阵限制在由双随机矩阵构成的流形内。由于这些矩阵的行和列和均为 1，操作 $\mathcal{H}_l^{\text{res}} \mathbf{x}_l$ 表现为输入特征的凸组合。这一特性促进了良好的信号传播，其中特征均值得以保持，信号范数被严格正则化，从而有效缓解了信号消失或爆炸的风险。此外，由于双随机矩阵在矩阵乘法下具有封闭性，复合映射 $\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\text{res}}$ 保留了这种守恒性质。因此，mHC 能有效维持任意深度之间的恒等映射稳定性。

# 二、预备知识

## （一）符号约定

在 HC 中，第 $l$ 层的输入为 $\mathbf{x}_l \in \mathbb{R}^{1 \times C}$，它被扩展 $n$ 倍，以构造一个隐状态矩阵 $\mathbf{x}_l = (\mathbf{x}_{l,0}^\mathrm{T}, \ldots, \mathbf{x}_{l,n-1}^\mathrm{T})^\mathrm{T} \in \mathbb{R}^{n \times C}$，这可以被视为 $n$ 条残差流，该操作有效地拓宽了残差流的宽度。为了控制该流的读出、写入和更新过程，HC 引入了三个可学习的线性映射：$\mathcal{H}_l^{\mathrm{pre}}, \mathcal{H}_l^{\mathrm{post}} \in \mathbb{R}^{1 \times n}$ 以及 $\mathcal{H}_l^{\mathrm{res}} \in \mathbb{R}^{n \times n}$。这些映射修改了公式 $\eqref{eq:s_res}$ 中所示的标准残差连接，从而得到 $\eqref{eq:hc}$ 中的公式形式。

HC 的可学习映射由两部分系数组成：输入依赖部分和全局部分，分别称为动态映射和静态映射。形式上，HC 按如下方式计算这些系数：

$$
\begin{equation}
\begin{aligned}
\begin{cases}
\tilde{\mathbf{x}}_l = \mathrm{RMSNorm}(\mathbf{x}_l) & \in \mathbb{R}^{n \times C} \\
\mathcal{H}_l^{\mathrm{pre}} = \alpha_l^{\mathrm{pre}} \cdot \tanh(\theta_l^{\mathrm{pre}} \tilde{\mathbf{x}}_l^\mathrm{T}) + \mathbf{b}_l^{\mathrm{pre}} & \in \mathbb{R}^{1 \times n} \\
\mathcal{H}_l^{\mathrm{post}} = \alpha_l^{\mathrm{post}} \cdot \tanh(\theta_l^{\mathrm{post}} \tilde{\mathbf{x}}_l^\mathrm{T}) + \mathbf{b}_l^{\mathrm{post}} & \in \mathbb{R}^{1 \times n} \\
\mathcal{H}_l^{\mathrm{res}} = \alpha_l^{\mathrm{res}} \cdot \tanh(\theta_l^{\mathrm{res}} \tilde{\mathbf{x}}_l^\mathrm{T}) + \mathbf{b}_l^{\mathrm{res}} & \in \mathbb{R}^{n \times n}
\end{cases}
\end{aligned}
\end{equation}
$$
其中 $\mathrm{RMSNorm}(\cdot)$ 应用于最后一个维度，**标量** $\alpha_l^{\mathrm{pre}}, \alpha_l^{\mathrm{post}}, \alpha_l^{\mathrm{res}} \in \mathbb{R}$ 是可学习的门控因子，初始化为较小的值。动态映射通过**线性投影**得到，其参数由 $\theta_l^{\mathrm{pre}}, \theta_l^{\mathrm{post}} \in \mathbb{R}^{1 \times C}$ 和 $\theta_l^{\mathrm{res}} \in \mathbb{R}^{n \times C}$ 定义；而静态映射则由可学习的偏置项 $\mathbf{b}_l^{\mathrm{pre}}, \mathbf{b}_l^{\mathrm{post}} \in \mathbb{R}^{1 \times n}$ 和 $\mathbf{b}_l^{\mathrm{res}} \in \mathbb{R}^{n \times n}$ 表示。值得注意的是，引入这些映射带来的计算开销可以忽略不计，因为典型的扩展率 $n$（例如 4）远小于输入维度 $C$。

## （二）数值不稳定性

尽管残差映射 $\mathcal{H}_l^{\mathrm{res}}$ 对性能至关重要，但其连续应用会对数值稳定性构成显著风险。 HC 跨越多个层扩展时，从第 $l$ 层到第 $L$ 层的有效信号传播由复合映射 $\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\mathrm{res}}$ 控制。由于可学习映射 $\mathcal{H}_l^{\mathrm{res}}$ 无约束，该复合映射不可避免地偏离恒等映射。因此，在前向传播和反向传播过程中，信号幅度容易出现爆炸或消失。如下图所示，以 mHC 为基线，HC 在约 12k 步时表现出意外的损失激增，这与梯度范数的不稳定性高度相关。

<div align="center"><img src="https://wkqpicture.oss-cn-beijing.aliyuncs.com/img/20260326223959113.png" width="80%" alt="image-20260326223951900"></div>

此外，对 $\mathcal{H}_l^{\mathrm{res}}$ 的分析验证了这种不稳定的机制。为了量化复合映射 $\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\mathrm{res}}$ 沿残差流如何放大信号，文章采用两个指标：第一个基于复合映射**行和**的最大绝对值，捕捉**前向传播**中的最坏情况扩张；第二个基于**列和**的最大绝对值，对应于**反向传播**。将这些指标称为复合映射的 Amax 增益幅度。如下图所示，Amax 增益幅度达到极端值，峰值高达 3000，与 1 相比存在巨大偏差，证实了残差流爆炸的存在。

<div align="center"><img src="https://wkqpicture.oss-cn-beijing.aliyuncs.com/img/20260326225513226.png" width="80%" alt="image-20260326225513048"></div>

# 三、方法

## （一）mHC

受恒等映射原则的启发，mHC 的核心思想是将残差映射 $\mathcal{H}_l^{\mathrm{res}}$ 投影到一个特定的流形上。虽然原始的恒等映射通过强制 $\mathcal{H}_l^{\mathrm{res}} = \mathbf{I}$ 来保证稳定性，但它从根本上阻止了残差流内部的信息交换，而这种交换对于充分发挥多流架构的潜力至关重要。因此，文章提出将残差映射投影到一个流形上，该流形既能保持跨层信号传播的稳定性，又能促进残差流之间的相互作用，从而保留模型的表达能力。为此，将 $\mathcal{H}_l^{\mathrm{res}}$ 限制为**双随机矩阵**，即其元素非负，且每行和每列之和均为 1。形式上，令 $\mathcal{M}^{\mathrm{res}}$ 表示**双随机矩阵的流形**（也称为 **Birkhoff 多面体**）。将 $\mathcal{H}_l^{\mathrm{res}}$ 限制在 $\mathcal{P}_{\mathcal{M}^{\mathrm{res}}}(\mathcal{H}_l^{\mathrm{res}})$ 上，定义为：

$$
\begin{equation}
\mathcal{P}_{\mathcal{M}^{\mathrm{res}}}(\mathcal{H}_l^{\mathrm{res}}) := \left\{ \mathcal{H}_l^{\mathrm{res}} \in \mathbb{R}^{n \times n} \mid \mathcal{H}_l^{\mathrm{res}} \mathbf{1}_n = \mathbf{1}_n,\ \mathbf{1}_n^\mathrm{T} \mathcal{H}_l^{\mathrm{res}} = \mathbf{1}_n^\mathrm{T},\ \mathcal{H}_l^{\mathrm{res}} \geqslant 0 \right\}
\end{equation}
$$

其中 $\mathbf{1}_n$ 表示全为 1 的 $n$ 维向量。

值得注意的是，当 $n = 1$ 时，双随机条件退化为标量 1，从而恢复原始的恒等映射。选择双随机性赋予了多个严格的理论性质，对大规模模型训练具有显著优势：

1. **范数保持性**：双随机矩阵的谱范数被限制在 1 以内（即 $\|\mathcal{H}_l^{\mathrm{res}}\|_2 \leq 1$）。这意味着可学习映射是非扩张的，有效缓解了梯度爆炸问题。

2. **组合封闭性**：双随机矩阵集合在矩阵乘法下是封闭的。这保证了跨多层的复合残差映射 $\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\mathrm{res}}$ 仍为双随机矩阵，从而在整个模型深度中保持稳定性。

3. **几何解释**：集合 $\mathcal{M}^{\mathrm{res}}$ 构成 Birkhoff 多面体，即置换矩阵集合的凸包。这提供了清晰的几何解释：残差映射可以看作是置换矩阵的凸组合。从数学上看，此类矩阵的重复应用会单调地增强流之间的信息混合，有效充当鲁棒的特征融合机制。

此外，对输入映射 $\mathcal{H}_l^{\mathrm{pre}}$ 和输出映射 $\mathcal{H}_l^{\mathrm{post}}$ 施加非负性约束。该约束防止了正负系数组合导致的信号抵消，也可视为一种特殊的流形投影。

---

>  如何理解上述第 3 点？

（1）**Birkhoff 多面体 (Birkhoff Polytope)**

在数学上，Birkhoff 多面体是指所有 **$n \times n$ 双随机矩阵**构成的集合。为什么叫“多面体”？如果你把一个 $n \times n$ 矩阵展平成一个 $n^2$ 维的向量，所有满足上述条件的矩阵在这个高维空间中会形成一个几何形状。这个形状是**有边界的**、**凸的**，因此被称为“多面体”（Polytope）。在 mHC 中，$\mathcal{M}^{\mathrm{res}}$ 就是这个多面体。论文强制要求学习到的连接矩阵 $\mathcal{H}_l^{\mathrm{res}}$ 必须落在这个多面体内部或表面上。

（2）**置换矩阵集合的凸包**

这是 **Birkhoff-von Neumann 定理** 的核心结论。要理解这句话，需要先理解两个子概念：

a. 置换矩阵 (Permutation Matrix)

置换矩阵是一种特殊的“硬”矩阵，它是双随机矩阵的**极端情况**。特点是元素只能是 0 或 1，且每行每列恰好有一个 1。它代表**硬路由（Hard Routing）**。例如：
$$
P = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
$$
这表示输入 $[x_1, x_2]$ 变成了 $[x_2, x_1]$。

b. 凸包 (Convex Hull)

想象平面上有几个点（代表所有的置换矩阵）。凸包就是把这些点用最紧的橡皮筋包起来形成的形状内部的所有点。
*   **数学定义**：集合中任意点的**凸组合**。
*   **凸组合 (Convex Combination)**：指加权平均，权重非负且和为 1。$H = \sum_{k} \theta_k P_k$，其中，$\theta_k \ge 0, \quad \sum \theta_k = 1$，这里 $P_k$ 是置换矩阵，$\theta_k$ 是权重。

Birkhoff 多面体是置换矩阵集合的凸包意味着：**任何一个双随机矩阵，都可以分解为若干个置换矩阵的加权平均。**

---

## （二）参数化和流形投影

接下来介绍 mHC 中 $\mathcal{H}_l^{\mathrm{pre}}$、$\mathcal{H}_l^{\mathrm{post}}$ 和 $\mathcal{H}_l^{\mathrm{res}}$ 的计算过程。给定第 $l$ 层的输入隐藏矩阵 $\mathbf{x}_l \in \mathbb{R}^{n \times C}$，我们首先将其展平为向量 $\vec{\mathbf{x}}_l = \mathrm{vec}(\mathbf{x}_l) \in \mathbb{R}^{1 \times nC}$，以保留完整的上下文信息。接着，我们遵循原始 HC 的公式，得到动态映射和静态映射如下：

$$
\begin{equation}
\begin{aligned}
\begin{cases}
\vec{\mathbf{x}}_l' = \mathrm{RMSNorm}(\vec{\mathbf{x}}_l) & \in \mathbb{R}^{1 \times nC} \\
\tilde{\mathcal{H}}_l^{\mathrm{pre}} = \alpha_l^{\mathrm{pre}} \cdot (\vec{\mathbf{x}}_l' \varphi_l^{\mathrm{pre}}) + \mathbf{b}_l^{\mathrm{pre}} & \in \mathbb{R}^{1 \times n} \\
\tilde{\mathcal{H}}_l^{\mathrm{post}} = \alpha_l^{\mathrm{post}} \cdot (\vec{\mathbf{x}}_l' \varphi_l^{\mathrm{post}}) + \mathbf{b}_l^{\mathrm{post}} & \in \mathbb{R}^{1 \times n} \\
\tilde{\mathcal{H}}_l^{\mathrm{res}} = \alpha_l^{\mathrm{res}} \cdot \mathrm{mat}(\vec{\mathbf{x}}_l' \varphi_l^{\mathrm{res}}) + \mathbf{b}_l^{\mathrm{res}} & \in \mathbb{R}^{n \times n}
\end{cases}
\end{aligned}
\end{equation}
$$

其中 $\varphi_l^{\mathrm{pre}}, \varphi_l^{\mathrm{post}} \in \mathbb{R}^{nC \times n}$ 和 $\varphi_l^{\mathrm{res}} \in \mathbb{R}^{nC \times n^2}$ 是用于动态映射的线性投影，而 $\mathrm{mat}(\cdot)$ 是从 $\mathbb{R}^{1 \times n^2}$ 到 $\mathbb{R}^{n \times n}$ 的reshape函数。

随后，最终的约束映射通过以下方式获得：

$$
\begin{equation}
\begin{aligned}
\begin{cases}
\mathcal{H}_l^{\mathrm{pre}} = \sigma(\tilde{\mathcal{H}}_l^{\mathrm{pre}}) \\
\mathcal{H}_l^{\mathrm{post}} = 2\sigma(\tilde{\mathcal{H}}_l^{\mathrm{post}}) \\
\mathcal{H}_l^{\mathrm{res}} = \text{Sinkhorn-Knopp}(\tilde{\mathcal{H}}_l^{\mathrm{res}})
\end{cases}
\end{aligned}
\end{equation}
$$

其中 $\sigma(\cdot)$ 表示 Sigmoid 函数。$\text{Sinkhorn-Knopp}(\cdot)$ 算子首先通过指数运算使所有元素变为正数，然后进行迭代归一化过程，交替对行和列进行缩放，使其和为 1。具体而言，以正矩阵 $\mathbf{M}^{(0)} = \exp(\tilde{\mathcal{H}}_l^{\mathrm{res}})$ 作为起点，归一化迭代过程如下：

$$
\begin{equation}
\begin{aligned}
\mathbf{M}^{(t)} = \mathcal{T}_r\left(\mathcal{T}_c(\mathbf{M}^{(t-1)})\right)
\end{aligned}
\end{equation}
$$

其中 $\mathcal{T}_r$ 和 $\mathcal{T}_c$ 分别表示行和列归一化。当 $t_{\max} \to \infty$ 时，该过程收敛到一个双随机矩阵 $\mathcal{H}_l^{\mathrm{res}} = \mathbf{M}^{(t_{\max})}$，在实验中选择 $t_{\max} = 20$ 作为实际值迭代次数。

>  为什么这里 $\mathcal{H}_l^{\mathrm{pre}}$ 使用的是 $\sigma$ 而 $\mathcal{H}_l^{\mathrm{post}}$ 使用的是 $2\sigma$ ？
>
> 原文在进行消融实验时提到（Table 1）：When a specific mapping ($\mathcal{H}_l^{\mathrm{pre}}$, $\mathcal{H}_l^{\mathrm{post}}$, or $\mathcal{H}_l^{\mathrm{res}}$) is disabled, we employ a fixed mapping to maintain dimensional consistency: uniform weights of $1/n$ for $\mathcal{H}_l^{\mathrm{pre}}$, uniform weights of ones for $\mathcal{H}_l^{\mathrm{post}}$, and the identity matrix for $\mathcal{H}_l^{\mathrm{res}}$.
>
> 这意味着 $\mathcal{H}_l^{\mathrm{pre}}$ 和 $\mathcal{H}_l^{\mathrm{post}}$ 在角色定位上略有不同。前者更像是读入时各 stream 的加权，用于把 n 条 stream 聚合成一个 layer input，它是一个多到一的过程，$\sigma$ 使权重落在 0 到 1 之间，防止聚合爆炸。后者更像是把一个 layer output 写回各 stream，默认强度与标准 residual 一致，是一个一到多的过程，零初始化时，$2\sigma$ 会使得权重在 1 附近。

# 四、实验

## （一）实验设置

通过对语言模型进行预训练来验证所提出的方法，对基线模型、HC 和 mHC 进行了对比分析。采用受 DeepSeek-V3 启发的 MoE 架构，训练了四种不同的模型变体以覆盖不同的评估场景，HC 和 mHC 的扩展率 $n$ 均设为 4。

## （二）主要结果

通过观察 27B 模型的训练稳定性、收敛情况和下游任务表现，发现：

- **训练稳定性**：mHC 有效缓解了 HC 的损失突增和梯度范数不稳定问题，最终损失比基线低 0.021。
- **下游性能**：在 8 个基准测试中，mHC 全面超越基线，并在大多数任务上优于 HC。

<div align="center"><img src="https://wkqpicture.oss-cn-beijing.aliyuncs.com/img/20260329094308627.png" width="80%" alt="image-20260329094301456"></div>

<div align="center"><img src="https://wkqpicture.oss-cn-beijing.aliyuncs.com/img/20260329094342940.png" width="80%" alt="image-20260329094342849"></div>

## （三）扩展性实验

- **计算规模扩展**：从 3B 到 27B 模型，mHC 相对于基线的性能优势随计算预算增加而保持。
- **数据规模扩展**：在 3B 模型的训练过程中，mHC 始终优于基线。

<div align="center"><img src="https://wkqpicture.oss-cn-beijing.aliyuncs.com/img/20260329094637990.png" width="80%" alt="image-20260329094637884"></div>

## （四）稳定性分析

- **传播稳定性**：mHC 将复合映射的最大增益幅度从 HC 的约 3000 降至约 1.6。mHC 的映射结果比 HC 的极端值更稳定。

<div align="center"><img src="https://wkqpicture.oss-cn-beijing.aliyuncs.com/img/20260329095229973.png" width="80%" alt="20260329095229973.png"></div>

# 五、总结

mHC 是 HC 的一个通用扩展框架。它通过 Sinkhorn-Knopp 算法将残差映射约束在双随机矩阵流形上，成功恢复了恒等映射特性，解决了 HC 在大规模训练中的不稳定性和可扩展性问题。结合高效的系统级优化，mHC 在带来性能提升的同时，仅引入极小的额外开销。
{% endraw %}