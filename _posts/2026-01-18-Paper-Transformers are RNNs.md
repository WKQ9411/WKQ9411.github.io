---
layout: post
title: 【论文解读】Transformers are RNNs
date: 2026-01-18 16:00:00 +08:00
summary: 本文提出了经典的 Linear Attention，大幅降低原始 Transformer 的内存与计算成本，利用矩阵乘积结合律使自注意力的时间和内存随序列长度呈线性增长。
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

Transformer 在多项任务中表现出色，但因其对输入序列长度的二次复杂度计算，在处理极长序列时速度过慢。为解决此问题，本文将自注意力表示为核特征映射的线性点积，利用矩阵乘法的结合律，将计算复杂度从 $O (N^2)$ 降至 $O (N)$，大幅加速了自回归 Transformer，并揭示其与循环神经网络（RNN）的关联。

原文链接：[Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)

# 一、Transformers

设 $x \in \mathbb{R}^{N \times F}$ 表示一个由 $N$ 个维度为 $F$ 的特征向量组成的序列。Transformer 是一个由 $L$ 个 transformer 层 $T_1(\cdot), \ldots, T_L(\cdot)$ 组成的函数 $T : \mathbb{R}^{N \times F} \to \mathbb{R}^{N \times F}$，每层定义如下：

$$
\begin{equation}
T_l(x) = f_l(A_l(x) + x) \label{eq:transformer}
\end{equation}
$$
其中，$f_l(\cdot)$ 独立地变换每个特征，通常通过一个小的两层前馈网络实现。$A_l(\cdot)$ 是自注意力函数，是 Transformer 中唯一跨序列作用的部分。

输入序列 $x$ 通过三个矩阵 $W_Q \in \mathbb{R}^{F \times D}$、$W_K \in \mathbb{R}^{F \times D}$ 和 $W_V \in \mathbb{R}^{F \times M}$ 投影为对应的表示 $Q$、$K$ 和 $V$。所有位置的输出 $A_l(x) = V'$ 计算如下：

$$
\begin{equation}
\begin{aligned}
Q &= xW_Q, \\
K &= xW_K, \\
V &= xW_V, \\
A_l(x) &= V' = \text{softmax}\left(\frac{QK^{\top}}{\sqrt{D}}\right)V
\end{aligned} \label{eq:softmax}
\end{equation}
$$
公式 $\eqref{eq:softmax}$ 即 softmax 注意力，它的相似度得分是 query 和 key 的点积的指数。进一步，我们可以使用任意相似度函数，写出一个通用的注意力公式如下：

$$
\begin{equation}
V'_i = \frac{\sum_{j=1}^{N} \text{sim}(Q_i, K_j) V_j}{\sum_{j=1}^{N} \text{sim}(Q_i, K_j)} \label{eq:common}
\end{equation}
$$

当将相似度函数替换为 $\text{sim}(q, k) = \exp\left(\frac{q^\top k}{\sqrt{D}}\right)$ 时，公式 $\eqref{eq:common}$ 等价于公式 $\eqref{eq:softmax}$。

> 论文中提到矩阵的下标 $i$ 表示取该矩阵第 $i$ 行的行向量，但当公式中写 $Q_i、K_i$ 作为向量时，是按照常用的列向量来理解的，因此会看到矩阵的 $QK^{\top}$ 和向量的 $q^{\top} k$、$\phi(Q_i)^{\top} \phi(K_j)$ 两种转置形式。

# 二、线性化注意力

为了让公式 $\eqref{eq:common}$ 能够定义为一个注意力函数，唯一需要的约束是 $\text{sim}(\cdot)$ 必须是非负的，这包括了所有的核函数 $k(x,y):\mathbb{R^{2 \times F}} \to \mathbb{R}_+$ 。

> 核函数 $k(x,y)$ 的本质，是某个（可能是高维甚至无限维）特征空间中，特征映射 $\phi(x),\phi(y)$ 的内积：$k(x,y)=\langle \phi(x),\phi(y) \rangle$ 。

给定一个具有特征表示 $\phi(x)$ 的核函数，我们可以将公式 $\eqref{eq:common}$ 改写为：
$$
\begin{equation}
V'_i = \frac{\sum_{j=1}^{N} \phi(Q_i)^{\top} \phi(K_j) V_j}{\sum_{j=1}^{N} \phi(Q_i)^{\top} \phi(K_j)}
\end{equation}
$$
根据矩阵乘法的结合律，可进一步写为：
$$
\begin{equation}
V'_i = \frac{\phi(Q_i)^{\top} \sum_{j=1}^{N} \phi(K_j) V_j^{\top}}{\phi(Q_i)^{\top} \sum_{j=1}^{N} \phi(K_j)} \label{eq:linear}
\end{equation}
$$
分子写为如下的矩阵形式更容易理解：
$$
\begin{equation}
\left(\phi(Q) \phi(K)^{\top}\right) V = \phi(Q) \left(\phi(K)^{\top} V\right)
\end{equation}
$$
其中，特征映射 $\phi(\cdot)$ 是逐行应用于矩阵 $Q$ 和 $K$ 的。

从公式 $\eqref{eq:softmax}$ 可以看出，softmax 注意力的计算和内存复杂度随序列长度 $N$ 呈 $O(N^2)$ 的规模增长。相比之下，线性注意力 $\eqref{eq:linear}$ 具有 $O(N)$ 的计算和内存复杂度，因为我们只需一次性计算 $\sum_{j=1}^{N} \phi(K_j) V_j^{\top}$ 和 $\sum_{j=1}^{N} \phi(K_j)$，然后对每个查询重复使用这些结果即可。

**特征映射与计算成本**

（1）对于 softmax 注意力，计算过程可以分为两个主要的矩阵乘法步骤：

- 计算注意力分数 ($QK^{\top}$)，矩阵 $Q$ 的维度是 $N \times D$，矩阵 $K^{\top}$ 的维度是 $D \times N$，计算开销为：$O(N^2 D)$。
- 计算最终输出（$\text{Attention} \times V$)，注意力矩阵（经过 softmax 后）维度是 $N \times N$，矩阵 $V$ 的维度是 $N \times M$，计算开销为：$O(N^2 M)$。
- 从量级上可以简记为 $O(N^2 \max(D, M))$。

（2）对于线性注意力：

- 首先将维度为 $D$ 的原始向量通过特征映射函数 $\phi(\cdot)$ 映射到新空间，维度记为 $C$。
- 计算顺序利用结合律发生了改变：$\phi(Q)(\phi(K)^{\top} V)$ ，先计算 $\phi(K)^{\top} V$，得到 $C \times M$ 的隐状态矩阵，计算开销为： $O(NCM)$，再用 $\phi(Q)$ 乘以该矩阵，计算开销相同，为 $O(NCM)$。

（3）对于一个简单的二阶齐次多项式核，其定义为：$k(x, y) = (x^{\top} y)^2$，其中 $x, y \in \mathbb{R}^D$ 是 $D$ 维向量。

为了找到显式的特征映射 $\phi(x)$，我们需要将上述标量乘积的平方展开。假设 $x = [x_1, x_2, \dots, x_D]^{\top}$ 和 $y = [y_1, y_2, \dots, y_D]^{\top}$：

- 点积展开： $x^{\top} y = \sum_{i=1}^D x_i y_i$
- 平方展开： $(x^{\top} y)^2 = (\sum_{i=1}^D x_i y_i) \cdot (\sum_{j=1}^D x_j y_j) = \sum_{i=1}^D \sum_{j=1}^D (x_i x_j)(y_i y_j)$
- 展开式中的每一项都是 $(x_i x_j)$ 与 $(y_i y_j)$ 的乘积，它是 $\phi(x)$ 的内积由于 $i$ 可以取 $1$ 到 $D$，$j$ 也可以取 $1$ 到 $D$，所有可能的组合坐标 $(i, j)$ 共有 $D \times D = D^2$ 个，即此时特征空间的维度为 $C=D^2$。
- 因此二阶多项式线性 Transformer 的复杂度即为 **$O(ND^2M)$** 。
- 此时模型在序列长度 $N$ 远大于 $D^2$ 时（$N > D^2$）具有显著的计算优势。

线性注意力的核心在于寻找一个合适的特征映射函数 $\phi(x)$，然而，这里存在一个理论上的挑战：标准 softmax 注意力使用的是指数核（Exponential Kernel），它对应的特征映射 $\phi(x)$ 是无穷维的，这意味着要精确线性化 Softmax 是不可行的。

为了处理论文实验中规模较小的序列，作者并未使用复杂的多项式核，而是采用了一种更简洁的特征映射 ：
$$
\begin{equation}
\phi(x) = \text{elu}(x) + 1
\end{equation}
$$
 这是为了确保相似度分数非负，elu (exponential linear unit) 激活函数表达如下：
$$
\begin{equation}
\text{elu}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \le 0 \end{cases}
\end{equation}
$$
其中 $\alpha$ 通常取 1.0。这样的 $\phi(x)$ 保证了 $x$ 为负数时仍有梯度，且始终非负，在此定义下有 $C=D$。

# 三、因果掩码

Transformer 架构可以通过对注意力计算进行掩码处理，来高效训练自回归模型，使得第 $i$ 个位置只能受到第 $j$ 个位置的影响，当且仅当 $j \le i$。形式上，这种因果掩码对公式 $\eqref{eq:common}$ 的修改如下：
$$
\begin{equation}
V'_i = \frac{\sum_{j=1}^{\color{red}{i}} \text{sim}(Q_i, K_j) V_j}{\sum_{j=1}^{\color{red}{i}} \text{sim}(Q_i, K_j)}
\end{equation}
$$
进一步可以写为：
$$
\begin{equation}
V'_i = \frac{\phi(Q_i)^{\top} \sum_{j=1}^{\color{red}{i}} \phi(K_j) V_j^{\top}}{\phi(Q_i)^{\top} \sum_{j=1}^{\color{red}{i}} \phi(K_j)} \label{eq:mask}
\end{equation}
$$
定义：
$$
\begin{equation}
S_i = \sum_{j=1}^{i} \phi(K_j) V_j^{\top}
\end{equation}
$$

$$
\begin{equation}
Z_i = \sum_{j=1}^{i} \phi(K_j)
\end{equation}
$$

$S_i$ 可以理解为**注意力存储状态**，$Z_i$ 可以理解为**归一化存储状态**。它们可以通过增量的方式进行更新，对于第 $t$ 步的状态，有：
$$
\begin{equation}
\begin{cases}
S_t = S_{t-1} + \phi(K_t) V_t^{\top}, \\
Z_t = Z_{t-1} + \phi(K_t),
\end{cases}
\end{equation}
$$
这意味着计算每个时间步的复杂度是**常数级**的。公式 $\eqref{eq:mask}$ 可以进一步写为：
$$
\begin{equation}
V'_i = \frac{\phi(Q_i)^{\top} S_i}{\phi(Q_i)^{\top} Z_i} \label{eq:output}
\end{equation}
$$

## （一）梯度计算

> 本部分公式较多，重要节点的公式将通过方框框住，大部分内容来源于论文附录。

在任何深度学习框架中，对公式 $\eqref{eq:output}$ 的简单实现，是先前向，记录计算图/中间张量，然后再按图回传梯度。在这种实现下，框架为了能回传梯度，会把每一步的中间值 $S_i$ 都缓存下来。

如前所述，$D$ 是 query/key 的维度，$M$ 是 value 的维度，$N$ 是序列长度。缓存输入序列的 $\phi(K)$、$V$，大小分别是 $N \times D$、$N \times M$，如果每一步需缓存 $S_i$ 矩阵，则需额外缓存的大小为 $N \times D \times M$，这会限制长序列和更深层模型的性能。

为了避免缓存全部的 $S_i$ ，可以将梯度也改写为累计和（cumulative sums）的形式，从而使前向与反向中的因果线性注意力计算都具有线性时间和恒定缓存开销。

接下来，我们推导标量损失对公式 $\eqref{eq:mask}$ 的梯度。其中，分母和整条分式的梯度直接交给 autograd，因为他们只涉及向量前缀和，内存压力小。重点只推导分子的梯度，因为分子中有 $\sum_{j=1}^{i} \phi(K_j) V_j^{\top}$，这正是我们需要解决的避免缓存多步 $S_i$ 的地方。

首先简化符号，把 $Q,K$ 直接当做已经过特征 $\phi(\cdot)$ 映射的向量，因此**分子**（即未归一化输出）可以写成：
$$
\begin{equation}
\bar{V}_i = Q_i^{\top} \sum_{j=1}^{i} K_j V_j^{\top}
\end{equation}
$$
因此为了计算 $\nabla_{\bar{V}} \mathcal{L}$，我们需要计算 $\nabla_Q \mathcal{L}$、$\nabla_K \mathcal{L}$ 和 $\nabla_V \mathcal{L}$。我们首先把上式的某个分量（第 $e$ 个 value 维度）写成**标量形式**：
$$
\begin{equation}
\boxed{\bar{V}_{ie} = \sum_{d=1}^{D} Q_{id} \sum_{j=1}^{i} K_{jd} V_{je} = \sum_{d=1}^{D} \sum_{j=1}^{i} Q_{id} K_{jd} V_{je}} \label{eq:element}
\end{equation}
$$
即如下图所示，灰色部分代表第 $e$ 个维度的分量：

<div align="center"><img src="https://wkqpicture.oss-cn-beijing.aliyuncs.com/img/20260107143842.png" width="40%" alt="Linear Sequence Modeling Methods"></div>

### 1. 对Q的梯度

为了对 $Q$ 求梯度，可从对任意 $Q_{lt}$ 求梯度开始。其中，$l$ 是指第 $l$ 个 token，$t$ 是指第 $t$ 维特征，由于 $Q \in \mathbb{R}^{N \times D}$，因此 $Q_{lt}$ 就是矩阵 $Q$ 的第 $l$ 行第 $t$ 列的元素。这里我们求的是**标量**损失 $\mathcal{L}$ 对**标量** $Q_{lt}$ 的偏导，$\mathcal{L}$ 不是直接依赖 $Q_{lt}$ 的，它是通过中间量 $\bar{V}$ 依赖的。对固定位置 $l$，$\bar{V}$ 是一个 $M$ 维向量，包含的元素有 $\bar{V}_{l1}, \dots, \bar{V}_{lM}$。因此要遍历每一个中间元素，根据链式法则，有：
$$
\begin{equation}
\frac{\partial \mathcal{L}}{\partial Q_{lt}} = \sum_{e=1}^{M} \frac{\partial \mathcal{L}}{\partial \bar{V}_{le}} \frac{\partial \bar{V}_{le}}{\partial Q_{lt}} \label{eq:qlt_chain}
\end{equation}
$$

> 为什么上述链式法则只用对 $e$ 从 $1 \sim M$ 求和，而不用考虑其他中间变量 $\bar{V}_{ie},(i \neq l)$？即为什么不对 $i$ 求和？根据公式 $\eqref{eq:element}$，当 $i \neq l$ 时，$\bar{V}_{ie}$ 里只会包含 $Q_{id}$，不会包含 $Q_{lt}$，所以 $\frac{\partial \bar{V}_{ie}}{\partial Q_{lt}}=0,(i \neq l)$，因此整个链式法则里无需 $\sum_i$，只保留 $i=l$ 即可。这也是原文中所说的：$Q_{lt}$ only affects $\bar{V}_l$ 。

现在继续计算 $\frac{\partial \bar{V}_{le}}{\partial Q_{lt}}$，首先把 $i=l$ 代入公式 $\eqref{eq:element}$ 中，可得：
$$
\begin{equation}
\bar{V}_{le} = \sum_{d=1}^{D} \sum_{j=1}^{l} Q_{ld} K_{jd} V_{je}
\end{equation}
$$
注意到当 $\frac{\partial Q_{ld}}{\partial Q_{lt}}=1$，当且仅当 $d=t$ 时，否则为 0。因此：
$$
\begin{equation}
\frac{\partial \bar{V}_{le}}{\partial Q_{lt}} = \sum_{d=1}^{D} \sum_{j=1}^{l} \frac{\partial (Q_{ld} K_{jd} V_{je})}{\partial Q_{lt}} = \sum_{j=1}^{l} K_{jt} V_{je}
\end{equation}
$$
因此，对任意 $Q_{lt}$ 求梯度有：
$$
\begin{equation}
\boxed{\frac{\partial \mathcal{L}}{\partial Q_{lt}} = \sum_{e=1}^{M} \frac{\partial \mathcal{L}}{\partial \bar{V}_{le}} \frac{\partial \bar{V}_{le}}{\partial Q_{lt}} = \sum_{e=1}^{M} \frac{\partial \mathcal{L}}{\partial \bar{V}_{le}} \left( \sum_{j=1}^{l} K_{jt} V_{je} \right)} \label{eq:qtl}
\end{equation}
$$
下面，需要把元素级的上式进一步转换为矩阵表达，并表示出它和前缀矩阵 $S_l = \sum_{j \le l} K_j V_j^{\top}$ 的关系。首先，引入一个上游梯度向量（维度为 $M$）：
$$
\begin{equation}
g_l \triangleq \nabla_{\bar{V}_l} \mathcal{L} \in \mathbb{R}^M, \quad (g_l)_e = \frac{\partial \mathcal{L}}{\partial \bar{V}_{le}}
\end{equation}
$$
可见该梯度向量的第 $e$ 个分量就是元素级的 $\frac{\partial \mathcal{L}}{\partial \bar{V}_{le}}$。此外，引入前缀矩阵（维度为 $D \times M$）：
$$
\begin{equation}
S_l \triangleq \sum_{j=1}^{l} K_j V_j^\top \in \mathbb{R}^{D \times M}
\end{equation}
$$
它的第 $t$ 行第 $e$ 列是：
$$
\begin{equation}
(S_l)_{te} = \sum_{j=1}^{l} K_{jt} V_{je}
\end{equation}
$$
那么公式 $\eqref{eq:qtl}$ 可以改写为：
$$
\begin{equation}
\frac{\partial \mathcal{L}}{\partial Q_{lt}} = \sum_{e=1}^{M} (g_l)_e (S_l)_{te} = g_l^{\top} \cdot (S_l)_t^{\top} \label{eq:dot}
\end{equation}
$$
可见，它实际上是向量 $g_l$ 与矩阵 $S_l$ 的第 $t$ 个行向量的点积（按维度 $e$ 求和）。现在我们想要的不是单个元素，而是整行向量的梯度：
$$
\begin{equation}
\nabla_{Q_l} \mathcal{L} = \left[ \frac{\partial \mathcal{L}}{\partial Q_{l1}}, \frac{\partial \mathcal{L}}{\partial Q_{l2}}, \ldots, \frac{\partial \mathcal{L}}{\partial Q_{lD}} \right] \in \mathbb{R}^D
\end{equation}
$$
对每个 $t$，我们都可用公式 $\eqref{eq:dot}$ 来计算，有：
$$
\begin{equation}
\nabla_{Q_l} \mathcal{L} = \left[ g_l^{\top}(S_l)_1^{\top}, g_l^{\top}(S_l)_2^{\top}, \ldots , g_l^{\top}(S_l)_D^{\top} \right] = g_l^{\top}S_l^{\top} \in \mathbb{R}^D
\end{equation}
$$
分别把 $S_l$ 和 $g_l$ 写回 $S_l = \sum_{j=1}^{l} K_j V_j^\top$ 和 $\nabla_{\bar{V}_l} \mathcal{L}$ ，将下标替换为 $i$，最终得到：
$$
\begin{equation}
\boxed{\nabla_{Q_i} \mathcal{L} = \nabla_{\bar{V}_i} \mathcal{L} \left( \sum_{j=1}^{l} K_j V_j^\top \right)^\top}
\end{equation}
$$

### 2. 对K的梯度

同样从对某个具体的 key 元素 $K_{lt}$ 开始计算梯度，但与公式 $\eqref{eq:qlt_chain}$ 中的 $Q_{lt}$ 只影响 $\bar{V}_l$ 不同，$K_l$ 会影响所有后续位置的前缀和。为方便起见，将公式 $\eqref{eq:element}$ 重新列出：
$$
\begin{equation}
\bar{V}_{ie} = \sum_{d=1}^{D} \sum_{j=1}^{i} Q_{id} K_{jd} V_{je}
\end{equation}
$$
可见在 $\bar{V}_i$ 里有 $\sum_{j=1}^{i}$，因此：

- 只要 $i \geq l$，前缀 $\{1, \ldots, i\}$ 就包含 $j = l$，于是 $\bar{V}_i$ 里就会出现 $K_{lt}$
- 当 $i < l$ 时，前缀不包含 $l$，所以 $K_{lt}$ 根本不会影响 $\bar{V}_i$

所以 $K_{lt}$ 会影响所有 $i = l, l+1, \ldots, N$ 的输出 $\bar{V}_i$，根据链式法则，有：
$$
\begin{equation}
\frac{\partial \mathcal{L}}{\partial K_{lt}} = \sum_{e=1}^{M} \sum_{i=1}^{N} \frac{\partial \mathcal{L}}{\partial \bar{V}_{ie}} \frac{\partial \bar{V}_{ie}}{\partial K_{lt}}
\end{equation}
$$
由于当 $i < l$ 时 $\frac{\partial \bar{V}_{ie}}{\partial K_{lt}} = 0$，因此进一步有：
$$
\begin{equation}
\frac{\partial \mathcal{L}}{\partial K_{lt}} = \sum_{e=1}^{M} \sum_{i=l}^{N} \frac{\partial \mathcal{L}}{\partial \bar{V}_{ie}} \frac{\partial \bar{V}_{ie}}{\partial K_{lt}}
\end{equation}
$$
现在算核心项 $\frac{\partial \bar{V}_{ie}}{\partial K_{lt}}$，从公式 $\eqref{eq:element}$ 中对 $K_{lt}$ 求偏导，只有当索引满足 $j = l$ 且 $d = t$ 时，项中才会出现 $K_{lt}$，所以：
$$
\begin{equation}
\frac{\partial \bar{V}_{ie}}{\partial K_{lt}} = \sum_{d=1}^{D} \sum_{j=1}^{i} Q_{id} V_{je} \frac{\partial K_{jd}}{\partial K_{lt}}
\end{equation}
$$
其中：
$$
\begin{equation}
\frac{\partial K_{jd}}{\partial K_{lt}} = 
\begin{cases}
1, & j = l \text{ and } d = t \\
0, & \text{otherwise}
\end{cases}
\end{equation}
$$
因此只剩下一个命中项：
$$
\begin{equation}
\frac{\partial \bar{V}_{ie}}{\partial K_{lt}} = Q_{it} V_{le} \quad (i \geq l).
\end{equation}
$$
把它代回链式法则，就得到：
$$
\begin{equation}
\frac{\partial \mathcal{L}}{\partial K_{lt}} = \sum_{e=1}^{M} \sum_{i=l}^{N} \frac{\partial \mathcal{L}}{\partial \bar{V}_{ie}} (Q_{it} V_{le})
\end{equation}
$$
同样的，引入上游梯度向量（维度为 $M$）：
$$
\begin{equation}
g_i \triangleq \nabla_{\bar{V}_i} \mathcal{L} \in \mathbb{R}^M, \quad (g_i)_e = \frac{\partial \mathcal{L}}{\partial \bar{V}_{ie}}
\end{equation}
$$
则有：
$$
\begin{equation}
\frac{\partial \mathcal{L}}{\partial K_{lt}} = \sum_{e=1}^{M} \sum_{i=l}^{N} (g_i)_e \, Q_{it} \, V_{le}
\end{equation}
$$
首先把对 $e$ 的求和视为一个点积，其中跟 $e$ 有关的部分是：
$$
\begin{equation}
\sum_{e=1}^{M} (g_i)_e V_{le}
\end{equation}
$$
它是两个 $M$ 维向量的点积：
$$
\begin{equation}
g_i^\top V_l
\end{equation}
$$
因此进一步得到：
$$
\begin{equation}
\frac{\partial \mathcal{L}}{\partial K_{lt}} = \sum_{i=l}^{N} Q_{it} \left( g_i^\top V_l \right) \label{eq:klt}
\end{equation}
$$
同样的，我们需要的是整行梯度：
$$
\begin{equation}
\nabla_{K_l} \mathcal{L} = \left[ \frac{\partial \mathcal{L}}{\partial K_{l1}}, \frac{\partial \mathcal{L}}{\partial K_{l2}}, \ldots, \frac{\partial \mathcal{L}}{\partial K_{lD}} \right] \in \mathbb{R}^D
\end{equation}
$$
将公式 $\eqref{eq:klt}$ 代入后得到：
$$
\begin{equation}
\nabla_{K_l} \mathcal{L} = \sum_{i=l}^{N} Q_i (g_i^\top V_l)
\end{equation}
$$
其中 $Q_i \in \mathbb{R}^D$，$g_i^\top$ 是 $1 \times M$ 的行向量，$g_i^\top V_l$ 是标量，即右边是若干个 $D$ 维向量的加权和，结果仍是 $D$ 维向量。此外：
$$
\begin{equation}
Q_i (g_i^\top V_l) = (Q_i g_i^\top) V_l
\end{equation}
$$
其中外积 $Q_i g_i^\top$ 是一个 $D \times M$ 的矩阵。于是有：
$$
\begin{equation}
\sum_{i=l}^{N} Q_i (g_i^\top V_l) = \sum_{i=l}^{N} (Q_i g_i^\top) V_l = \left( \sum_{i=l}^{N} Q_i g_i^\top \right) V_l
\end{equation}
$$
这里的变换相当于是把 $V_l$ 提取了出来。把下标从 $l$ 改为 $i$，把求和索引从 $i$ 改为 $j$ 避免冲突，同时把 $g_j$ 写回 $\nabla_{\bar{V}_j} \mathcal{L}$，最终得到：
$$
\begin{equation}
\boxed{\nabla_{K_i} \mathcal{L} = \left( \sum_{j=i}^{N} Q_j (\nabla_{\bar{V}_j} \mathcal{L})^\top \right) V_i}
\end{equation}
$$
可以看到，关于 $Q$ 和 $K$ 的梯度累计和矩阵具有相同的大小 $D \times M$：

- 对 $Q_l$，梯度只来自相同位置 $l$，所以是前缀结构（$\sum_{j=1}^i$）
- 对 $K_i$，它影响所有未来位置 $j \geq i$，所以是后缀结构（$\sum_{j=i}^N$）

### 3. 对V的梯度

与对 $K$ 的推导类似，从 $V_{lt}$ 开始计算梯度。同样为方便起见，再次将公式 $\eqref{eq:element}$ 重新列出：
$$
\begin{equation}
\bar{V}_{ie} = \sum_{d=1}^{D} \sum_{j=1}^{i} Q_{id} K_{jd} V_{je}
\end{equation}
$$
相同的：

- 只要 $i \geq l$，前缀 $\{1, \ldots, i\}$ 就包含 $j = l$，于是 $\bar{V}_i$ 里就会出现 $V_{lt}$
- 当 $i < l$ 时，前缀不包含 $l$，所以 $V_{lt}$ 根本不会影响 $\bar{V}_i$

因此链式法则是：
$$
\begin{equation}
\frac{\partial \mathcal{L}}{\partial V_{lt}} = \sum_{e=1}^{M} \sum_{i=l}^{N} \frac{\partial \mathcal{L}}{\partial \bar{V}_{ie}} \frac{\partial \bar{V}_{ie}}{\partial V_{lt}}
\end{equation}
$$
可见，$K_l, V_l$ 都会影响所有的未来位置 $i \geq l$。

下面计算 $\frac{\partial \bar{V}_{ie}}{\partial V_{lt}}$，从公式 $\eqref{eq:element}$ 中对 $V_{lt}$ 求偏导，只有当索引满足 $j = l$ 且 $e = t$ 时，项中才会出现 $V_{lt}$，所以：
$$
\begin{equation}
\frac{\partial \bar{V}_{ie}}{\partial V_{lt}} = \sum_{d=1}^{D} \sum_{j=1}^{i} Q_{id} K_{jd} \frac{\partial V_{je}}{\partial V_{lt}}
\end{equation}
$$
其中：
$$
\begin{equation}
\frac{\partial V_{je}}{\partial V_{lt}} = 
\begin{cases}
1, & j = l \text{ and } e = t \\
0, & \text{otherwise}
\end{cases}
\end{equation}
$$
因此：
$$
\begin{equation}
\frac{\partial \bar{V}_{ie}}{\partial V_{lt}} = \left( \sum_{d=1}^{D} Q_{id} K_{ld} \right) = (Q_i^\top K_l), \quad (i \geq l)
\end{equation}
$$
把它代回链式法则，得到：
$$
\begin{equation}
\frac{\partial \mathcal{L}}{\partial V_{lt}} = \sum_{i=l}^{N} \frac{\partial \mathcal{L}}{\partial \bar{V}_{it}} (Q_i^\top K_l)
\end{equation}
$$
注意，这里在计算 $\frac{\partial V_{je}}{\partial V_{lt}}$ 时，已经将 $e$ 约束为了 $t$ 。和前一样，定义上游梯度向量：
$$
\begin{equation}
g_i \triangleq \nabla_{\bar{V}_i} \mathcal{L} \in \mathbb{R}^M, \quad (g_i)_t = \frac{\partial \mathcal{L}}{\partial \bar{V}_{it}}
\end{equation}
$$
进而得到：
$$
\begin{equation}
\frac{\partial \mathcal{L}}{\partial V_{lt}} = \sum_{i=l}^{N} (g_i)_t (Q_i^\top K_l) \label{eq:vlt}
\end{equation}
$$
我们需要的是整行梯度：
$$
\begin{equation}
\nabla_{V_l} \mathcal{L} = \left[ \frac{\partial \mathcal{L}}{\partial V_{l1}}, \frac{\partial \mathcal{L}}{\partial V_{l2}}, \ldots, \frac{\partial \mathcal{L}}{\partial V_{lM}} \right] \in \mathbb{R}^M
\end{equation}
$$
将公式 $\eqref{eq:vlt}$ 代入得到：
$$
\begin{equation}
\nabla_{V_l} \mathcal{L} = \sum_{i=l}^{N} g_i (Q_i^\top K_l)
\end{equation}
$$
类似的，为了提取出 $K_l$，可以进一步写为：
$$
\begin{equation}
\nabla_{V_l} \mathcal{L} = \sum_{i=l}^{N} g_i (Q_i^\top K_l) = \sum_{i=l}^{N} (g_i Q_i^\top) K_l = \left( \sum_{i=l}^{N} (Q_i g_i^\top)^\top \right) K_l
\end{equation}
$$
其中，外积 $Q_i g_i^\top$ 是一个 $D \times M$ 的矩阵。把下标从 $l$ 改为 $i$，把求和索引从 $i$ 改为 $j$ 避免冲突，同时把 $g_j$ 写回 $\nabla_{\bar{V}_j} \mathcal{L}$，最终得到：
$$
\begin{equation}
\boxed{\nabla_{V_i} \mathcal{L} = \left( \sum_{j=i}^{N} Q_j (\nabla_{\bar{V}_j} \mathcal{L})^\top \right)^\top K_i}
\end{equation}
$$

### 4. 小节

最终，我们将之前为了简化，省略去 $\phi(\cdot)$ 的表达恢复，得到最终计算的梯度如下：
$$
\begin{align}
\nabla_{\phi(Q_i)} \mathcal{L} &= \nabla_{\bar{V}_i} \mathcal{L} \left( \sum_{j=1}^{i} \phi(K_j) V_j^\top \right)^{\top} \label{eq:gq}\\
\nabla_{\phi(K_i)} \mathcal{L} &= \left( \sum_{j=i}^{N} \phi(Q_j) (\nabla_{\bar{V}_j} \mathcal{L})^\top \right) V_i \\
\nabla_{V_i} \mathcal{L} &= \left( \sum_{j=i}^{N} \phi(Q_j) (\nabla_{\bar{V}_j} \mathcal{L})^\top \right)^{\top} \phi(K_i) \label{eq:gv}
\end{align}
$$
其中，为了计算 $\nabla_{\phi(Q_i)} \mathcal{L}$，只需维护一个前缀矩阵，为了计算 $\nabla_{\phi(K_i)} \mathcal{L}$ 和 $\nabla_{V_i} \mathcal{L}$，只需维护一个后缀矩阵（两个矩阵相同，只是差了一个转置），避免了简单实现中需缓存所有的中间 $S_i$。结合公式 $\eqref{eq:output}$，最终可以同时在前向、反向中，都保持线性的计算时间和固定大小的缓存。前向和反向过程中，分子计算的伪代码如下所示：

<div align="center"><img src="https://wkqpicture.oss-cn-beijing.aliyuncs.com/img/20260108154219.png" width="50%" alt="前向与反向伪代码"></div>

## （二）训练和推理

在训练自回归 Transformer 模型时，完整的真实序列是可以获取的，这使得公式 $\eqref{eq:transformer}$ 中的 $f_l(\cdot)$ 和注意力计算都能够实现分层并行。因此，Transformer 模型的训练效率比 RNN 更高。另一方面，在推理过程中，时间步 $i$ 的输出会成为时间步 $i+1$ 的输入，这使得自回归模型无法进行并行化处理。此外，Transformer 模型每个时间步的成本并非固定不变的，而是与当前序列长度的平方成正比。

本文提出的线性 Transformer 兼具两者的优势。在训练方面，计算可以并行化，并充分利用GPU或其他加速器。在推理方面，每步预测的时间成本和内存成本都是恒定的。这意味着我们可以简单的将 $\phi(K)V^{\top}$ 存储为内部状态，并像 RNN 一样在每个时间步更新它，这使得推理速度比其他 Transformer 模型快数千倍。

# 四、Transformers are RNNs

通常，Transformer 模型被视为与 RNN 是两种根本不同的方法。但通过上面的讨论可知，任何带因果掩码的 Transformer 层可以表示为：给定输入，修改内部状态后再预测输出的模型，即 RNN。

通过如下的等式，我们将公式 $\eqref{eq:transformer}$ 中的 Transformer 层形式化为一个 RNN。由此得到的 RNN 具有两个隐藏状态，即 **attention memory** $s$ 和 **normalizer memory** $z$ 。我们使用下标来表示循环中的时间步：
$$
\begin{align}
s_0 &= 0 \\ 
z_0 &= 0 \\
s_i &= s_{i-1} + \phi(x_i W_K) (x_i W_V)^\top \\
z_i &= z_{i-1} + \phi(x_i W_K) \\
y_i &= f_l \left( \frac{\phi(x_i W_Q)^\top s_i}{\phi(x_i W_Q)^\top z_i} + x_i \right)
\end{align}
$$
在上述等式中，对特征函数没有施加任何约束，理论上它可以表示任何 Transformer 模型，包括使用 softmax 注意力的模型。这些公式揭示了 Transformer 和 RNN 之间的关系，是我们更好的理解信息存储与检索的过程。

原文代码链接：[https:// linear-transformers.com](https:// linear-transformers.com) ，其中公式 $\eqref{eq:gq}$ - $\eqref{eq:gv}$ 大约是通过 200 行 CUDA 代码实现的。

# 五、总结

文章的实验部分省略。本文提出了线性 Transformer 模型，大幅降低原始 Transformer 的内存与计算成本，利用矩阵乘积结合律使自注意力的时间和内存随序列长度呈线性增长，且在因果掩码下仍保持线性渐近复杂度。
{% endraw %}