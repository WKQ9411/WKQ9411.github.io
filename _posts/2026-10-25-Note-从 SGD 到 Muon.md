---
layout: post
title: 【笔记】从 SGD 到 Muon
date: 2026-10-25 16:00:00 +08:00
summary: 从 SGD 到 Muon 的优化器原理。
categories: Note
excerpt_separator: <!--more-->
---

{% raw %}
# 目录
{:.no_toc}

* TOC
{:toc}

---

# 一、梯度

模型训练的目标是最小化损失函数：

$$
\begin{equation}
\min_{\theta} L(\theta)
\end{equation}
$$

其中：

- $\theta$：模型参数，可以是标量、向量、矩阵，实际 LLM 里是一大堆参数张量；
- $L(\theta)$：loss，衡量模型预测和目标答案之间的差距；
- 优化器的任务：根据梯度决定参数怎么更新。

反向传播会计算出梯度：

$$
\begin{equation}
g_t = \nabla_{\theta} L(\theta_t)
\end{equation}
$$

梯度表示：在当前参数 $\theta_t$ 处，参数往哪个方向变化会让 loss 增加最快。所以最基础思想是：

$$
\begin{equation}
\theta_{t+1} = \theta_t - \eta g_t
\end{equation}
$$

其中：

- $\eta$：learning rate，学习率；
- $-g_t$：loss 下降最快的局部方向。

这就是 SGD 的核心。

## （一）标量参数的梯度

假设只有一个参数 $\theta \in \mathbb{R}$，loss 是 $L(\theta)$，梯度就是普通导数：

$$
\begin{equation}
g = \frac{dL}{d\theta}
\end{equation}
$$

如果 $g > 0$，说明 $\theta$ 增大会让 loss 增大，所以应该减小 $\theta$。如果 $g < 0$，说明 $\theta$ 增大会让 loss 减小，所以应该增大 $\theta$。统一写法就是：

$$
\begin{equation}
\theta \leftarrow \theta - \eta g
\end{equation}
$$

## （二）向量参数的梯度

模型参数如果是向量：

$$
\begin{equation}
\theta =
\begin{bmatrix}
\theta_1 \\
\theta_2 \\
\cdots \\
\theta_n
\end{bmatrix}
\end{equation}
$$

loss 是标量 $L(\theta) \in \mathbb{R}$，梯度是每个参数对 loss 的偏导组成的向量：

$$
\begin{equation}
\nabla_\theta L =
\begin{bmatrix}
\frac{\partial L}{\partial \theta_1} \\
\frac{\partial L}{\partial \theta_2} \\
\cdots \\
\frac{\partial L}{\partial \theta_n}
\end{bmatrix}
\end{equation}
$$

SGD 更新是**逐元素**的：

$$
\begin{equation}
\theta_i \leftarrow \theta_i - \eta \frac{\partial L}{\partial \theta_i}
\end{equation}
$$

写成向量形式：

$$
\begin{equation}
\theta_{t+1} = \theta_t - \eta g_t
\end{equation}
$$

## （三）矩阵参数的梯度

Transformer 里大量参数是矩阵，例如 Linear 层：

$$
\begin{equation}
y = xW
\end{equation}
$$

其中 $W \in \mathbb{R}^{d_{in} \times d_{out}}$，loss 对矩阵 $W$ 的梯度也是一个同形状矩阵：

$$
\begin{equation}
\nabla_W L =
\begin{bmatrix}
\frac{\partial L}{\partial W_{11}} & \cdots & \frac{\partial L}{\partial W_{1m}} \\
\vdots & \ddots & \vdots \\
\frac{\partial L}{\partial W_{n1}} & \cdots & \frac{\partial L}{\partial W_{nm}}
\end{bmatrix}
\end{equation}
$$

普通 SGD 对矩阵参数仍然是逐元素更新：

$$
\begin{equation}
W_{t+1} = W_t - \eta G_t
\end{equation}
$$

其中：

$$
\begin{equation}
G_t = \nabla_W L(W_t)
\end{equation}
$$

# 二、SGD

理论上，可以用整个训练集样本计算 loss：

$$
\begin{equation}
L(\theta) = \frac{1}{N}\sum_{i=1}^{N} L_i(\theta)
\end{equation}
$$

梯度是：

$$
\begin{equation}
g = \frac{1}{N}\sum_{i=1}^{N} \nabla_\theta L_i(\theta)
\end{equation}
$$

然后更新：

$$
\begin{equation}
\theta_{t+1} = \theta_t - \eta g
\end{equation}
$$

问题是，训练集太大，每一步都用全量数据太贵。实际训练中，每一步只采样一个 batch：

$$
\begin{equation}
B_t \subset \{1,2,\ldots,N\}
\end{equation}
$$

计算 batch loss：

$$
\begin{equation}
L_{B_t}(\theta) = \frac{1}{|B_t|}\sum_{i \in B_t} L_i(\theta)
\end{equation}
$$

得到近似梯度：

$$
\begin{equation}
g_t = \nabla_\theta L_{B_t}(\theta_t)
\end{equation}
$$

更新：

$$
\begin{equation}
\theta_{t+1} = \theta_t - \eta g_t
\end{equation}
$$

这就是通常说的 SGD，严格说是 mini-batch SGD。SGD 简单，但有几个问题：

1. 梯度噪声大。因为每次只看一个 batch，不同 batch 的梯度方向可能差异很大。
2. 不同参数的梯度尺度不同。某些参数梯度大，某些参数梯度小，用同一个学习率不一定合适。
3. loss 地形可能很狭长。例如某个方向坡很陡，另一个方向坡很缓，SGD 容易在陡峭方向震荡，在平缓方向前进很慢。

# 三、Momentum

## （一）基本思想

SGD 只看当前梯度：

$$
\begin{equation}
\theta_{t+1} = \theta_t - \eta g_t
\end{equation}
$$

Momentum 则会维护一个动量：

$$
\begin{equation}
m_t = \beta m_{t-1} + (1-\beta) g_t
\end{equation}
$$

其中：

- $m_t$：历史梯度的指数滑动平均；
- $\beta$：动量系数，常见如 0.9；
- $g_t$：当前 batch 的梯度。

更新：

$$
\begin{equation}
\theta_{t+1} = \theta_t - \eta m_t
\end{equation}
$$

为什么动量有用？把递推展开：
$$
\begin{equation}
m_t = (1-\beta)g_t + \beta(1-\beta)g_{t-1} + \beta^2(1-\beta)g_{t-2} + \cdots
\end{equation}
$$

所以 $m_t$ 是过去梯度的加权平均，越近的梯度权重越大，越远的梯度权重越小。如果多个 step 的梯度方向一致，它们会累积，更新变快。如果梯度方向来回震荡，它们会互相抵消，更新更稳定。

## （二）Nesterov Momentum

普通 Momentum 是先看当前位置梯度，再累积动量。Nesterov Momentum 的思想是：先沿着已有动量预走一步，再看那个位置的梯度。一种常见写法：
$$
\begin{align}
g_t &= \nabla_\theta L(\theta_t - \eta \beta m_{t-1}) \\
m_t &= \beta m_{t-1} + g_t \\
\theta_{t+1} &= \theta_t - \eta m_t
\end{align}
$$
它的直觉是：不只看当前位置，而是看按照当前惯性即将到达的位置，从而提前修正方向。

# 四、Adam

## （一）基本思想

Momentum 只平滑了梯度方向，但仍然所有参数共享同一个学习率。Adam 引入了两个统计量：

1. 一阶矩：梯度均值；
2. 二阶矩：梯度平方均值。

它想让不同参数有不同的自适应步长。

Adam 的一阶矩：
$$
\begin{equation}
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
\end{equation}
$$

这和 Momentum 类似，是梯度的滑动平均。常见 $\beta_1 = 0.9$。

Adam 的二阶矩：
$$
\begin{equation}
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
\end{equation}
$$

这里 $g_t^2$ 是逐元素平方，它记录的是每个参数历史梯度平方的滑动平均大小。常见 $\beta_2 = 0.999$。

## （二）Bias correction

一开始：

$$
\begin{equation}
m_0 = 0,\quad v_0 = 0
\end{equation}
$$

这意味着刚开始训练时，历史梯度的统计量都被认为是零，所以训练初期 $m_t, v_t$ 会偏小。

以一阶矩为例，展开递推公式有：
$$
\begin{equation}
\begin{aligned}
m_t &= \beta_1 m_{t-1}+(1-\beta_1)g_t \\
&= \beta_1^2m_{t-2} +\beta_1(1-\beta_1)g_{t-1}+(1-\beta_1)g_t \\
&= \beta_1^t m_0 + \beta_1^{t-1} (1-\beta_1) g_1 + \cdots + \beta_1(1-\beta_1)g_{t-1} + (1-\beta_1)g_t\\
&=(1-\beta_1)\sum_{i=1}^{t} \beta_1^{t-i}g_i
\end{aligned}
\end{equation}
$$
假设梯度在统计意义上保持稳定，即每一步梯度的期望均为 $\mathbb{E}[g_i]=\mu$，那么一阶矩的期望为：
$$
\begin{equation}
\begin{aligned}
\mathbb{E}[m_t] &=(1-\beta_1)\sum_{i=1}^{t} \beta_1^{t-i}\mathbb{E}[g_i]\\ &=(1-\beta_1^t)\mu
\end{aligned}
\end{equation}
$$
所以，$m_t$ 的期望被缩小了 $1-\beta_1^t$ 倍。既然知道缩小的比例，就可以将它除掉：
$$
\begin{equation}
\hat m_t=\frac{m_t}{1-\beta_1^t}
\end{equation}
$$
这样就有 $\mathbb{E}[\hat m_t]=\mu$。同理，对于二阶矩：
$$
\begin{equation}
\hat v_t=\frac{v_t}{1-\beta_2^t}
\end{equation}
$$
Adam 最终更新：

$$
\begin{equation}
\theta_{t+1}
=
\theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
= \theta_t - \frac{\eta}{\sqrt{\hat v_t}+\epsilon} \hat m_t
\end{equation}
$$

其中：

- 分子 $\hat m_t$：平滑后的梯度方向；
- 分母 $\sqrt{\hat v_t}$：历史梯度 RMS；
- $\epsilon$：防止除零，常见 $10^{-8}$；
- $\frac{\eta}{\sqrt{\hat v_t}+\epsilon}$ 相当于逐参数自适应学习率。

Adam 的每个参数更新大致是：历史平均梯度 / 历史梯度尺度。如果某个参数梯度尺度很大，分母大，步长被压小。如果某个参数梯度尺度很小，分母小，相对步长变大。

# 五、AdamW

## （一）Weight decay

Weight decay 是一种参数正则化手段，让参数不要无限变大。最朴素的做法是在 loss 里加 L2 正则：

$$
\begin{equation}
L'(\theta) = L(\theta) + \frac{\lambda}{2}\|\theta\|^2
\end{equation}
$$

对应梯度：

$$
\begin{equation}
\nabla_\theta L' = \nabla_\theta L + \lambda \theta
\end{equation}
$$

于是普通更新变成：

$$
\begin{equation}
\theta_{t+1}
=
\theta_t - \eta(g_t + \lambda \theta_t)
\end{equation}
$$

展开：

$$
\begin{equation}
\theta_{t+1}
=
(1-\eta\lambda)\theta_t - \eta g_t
\end{equation}
$$

这就是 weight decay 的直观含义：每一步把参数缩小一点。

## （二）基本思想

如果把 L2 正则直接加到 Adam 的梯度里：

$$
\begin{equation}
g'_t = g_t + \lambda \theta_t
\end{equation}
$$

Adam 更新会变成：

$$
\begin{equation}
\theta_{t+1}
=
\theta_t
-
\eta
\frac{\hat m'_t}{\sqrt{\hat v'_t}+\epsilon}
\end{equation}
$$

问题是 $\lambda \theta_t$ 也会被 Adam 的二阶矩分母缩放。这样 weight decay 的强度不再是统一的，而是被每个参数的历史梯度尺度影响。这不是理想的 weight decay。

AdamW 把 weight decay 从梯度里拿出来，单独作用在参数上：

$$
\begin{equation}
\theta_{t+1}
=
\theta_t(1-\eta\lambda)
-
\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\end{equation}
$$

称为 **decoupled weight decay**。这也是现在 LLM 训练里 AdamW 比 Adam 更常用的原因。AdamW 很强，但它本质上还是逐元素优化器。对于矩阵参数 $W \in \mathbb{R}^{n \times m}$，AdamW 关注的是每个元素 $W_{ij}$，它会为每个元素维护一阶矩和二阶矩 $m_{ij}, v_{ij}$，但它不直接关心整个矩阵的结构，比如：

- 行和列之间的相关性；
- 矩阵的谱范数；
- 奇异值分布；
- 某些方向是否被过度放大。

Transformer 里的大部分参数是矩阵，例如：

- attention 的 $W_Q, W_K, W_V, W_O$；
- MLP 的 up/gate/down projection；
- MoE expert 矩阵。

这些矩阵的整体结构会影响信号传播和训练稳定性。因此，就有了接下来的 Muon，它的出发点是：对矩阵参数，不只是逐元素自适应，而是对整个更新矩阵做结构化处理。在此之前，先回顾一下数学基础。

# 六、必要数学基础

## （一）矩阵与线性变换

矩阵可以看作一种线性变换，它将输入向量映射到输出向量，并可能改变向量的方向和长度。设矩阵 $A\in\mathbb{R}^{m\times n}$，输入向量 $x\in\mathbb{R}^n$，则：
$$
\begin{equation}
y=Ax,\quad y\in\mathbb{R}^m
\end{equation}
$$
从几何上看，矩阵变换可能包含旋转、反射、缩放、剪切等效果。理解矩阵的核心问题之一，是找到一组特殊的方向，使矩阵在这些方向上的作用尽可能简单。

特征分解与奇异值分解都可以帮助我们分析矩阵的方向结构和缩放特性。

## （二）特征值与特征向量

### 1. 定义

对于方阵 $A\in\mathbb{R}^{n\times n}$，如果存在非零向量 $v$ 和标量 $\lambda$，满足：
$$
\begin{equation}
Av=\lambda v \label{eq:eigenvalue equation}
\end{equation}
$$
则称：

- $v$ 为矩阵 $A$ 的特征向量；
- $\lambda$ 为对应的特征值。

这个公式表明，矩阵 $A$ 作用于特征向量 $v$ 后，输出仍然沿着原来的方向，只是长度发生了变化；如果特征值为负数，输出方向还会反转。特征向量描述矩阵变换中的特殊方向，特征值描述矩阵沿这些方向的缩放倍数。

### 2. 求特征值

由公式 $\eqref{eq:eigenvalue equation}$ 移项可得：
$$
\begin{equation}
(A-\lambda I)v=0
\end{equation}
$$
其中 $I$ 为单位矩阵。由于 $v\neq 0$，要使齐次线性方程存在非零解，必须满足：
$$
\begin{equation}
\det(A-\lambda I)=0
\end{equation}
$$
这个方程称为特征方程。求解它可以得到特征值，再将特征值代入式 $\eqref{eq:eigenvalue equation}$，求得相应的特征向量。

### 3. 特征值与矩阵变换的关系

如果矩阵 $A$ 有一组**线性无关**的特征向量，可以将它们排列成矩阵 $Q=[v_1,v_2,\ldots,v_n]$，将特征值排列成对角矩阵：
$$
\begin{equation}
\Lambda= \begin{bmatrix} \lambda_1&&0\\ &\ddots&\\ 0&&\lambda_n \end{bmatrix}
\end{equation}
$$
由 $Av_i=\lambda_i v_i$，可得：
$$
\begin{equation}
AQ=Q\Lambda
\end{equation}
$$
由以上假设可知 $Q$ 是可逆的，两边右乘 $Q^{-1}$，得到：
$$
\begin{equation}
A=Q\Lambda Q^{-1}
\end{equation}
$$
这就是特征分解。它将矩阵分解为三个部分：

- $Q^{-1}$：将向量转换到特征向量构成的坐标系；
- $\Lambda$：沿各个特征方向进行缩放；
- $Q$：将结果转换回原坐标系。

需要注意，特征分解要求矩阵是方阵，而且一般方阵不一定可以在实数范围内进行完整的特征分解。

如何理解 $Q^{-1}$ 的作用是将向量转换到特征向量构成的坐标系？假设在二维平面上有一个物理向量 $\vec{v}$，在默认的标准坐标系（基向量是 $e_1 = (1,0)^T$, $e_2 = (0,1)^T$）下，它的坐标是 $\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$。这意味着物理向量 $\vec{v} = x_1 \cdot e_1 + x_2 \cdot e_2$。现在，换一组新基，也就是特征向量 $q_1$ 和 $q_2$。在这个新视角下，它的坐标是 $\begin{pmatrix} y_1 \\ y_2 \end{pmatrix}$。这意味着同一个物理向量 $\vec{v} = y_1 \cdot q_1 + y_2 \cdot q_2$。

因此：
$$
\begin{equation}
\vec{v} = x = Q y
\end{equation}
$$
这个式子说明：矩阵 $Q$ 的作用，就是把新坐标 $y$ 转换回原来的物理向量 $\vec{v}$。即同一个物理向量 $\vec{v}$，在标准基下的坐标是 $x$，在新基下的坐标是 $y$。进一步得到：
$$
\begin{equation}
y = Q^{-1} x
\end{equation}
$$
因此，$Q^{-1}$ 作用后，表示用特征向量作为基底的坐标。然后在这个坐标系下，$\Lambda$ 对各个方向进行缩放，互不干扰。最后 $Q$ 的作用类似，将坐标转换为原坐标系。

### 4. 实对称矩阵的正交特征分解

特征分解中有一类特别重要的矩阵，实对称矩阵。若 $A^\top=A$，则 $A$ 是实对称矩阵。根据实对称矩阵的谱定理，它一定可以进行正交特征分解：
$$
\begin{equation}
A=Q\Lambda Q^\top
\end{equation}
$$
其中 $Q^\top Q=QQ^\top=I$，称 $Q$ 为正交矩阵（Orthogonal Matrix）。也就是说，$Q$ 的列向量两两正交，且长度均为 1，因此 $Q^{-1}=Q^\top$。这类矩阵的优势在于，特征向量可以构成一组标准正交基。

对于正交矩阵 $Q$，对任意向量 $x$，有：
$$
\begin{equation}
\begin{aligned}
\|Qx\|_2^2 &=(Qx)^\top(Qx)\\ &=x^\top Q^\top Qx\\ &=x^\top x\\ &=\|x\|_2^2
\end{aligned}
\end{equation}
$$
因此，正交矩阵不会改变向量的长度。它所执行的变换可以理解为旋转或反射。正交特征分解可以理解为：

1. 通过 $Q^\top$ 将输入转换到正交特征向量构成的坐标系。
2. 通过 $\Lambda$ 沿各个正交方向分别缩放。
3. 通过 $Q$ 将结果映射回原坐标系。

正交特征分解揭示了矩阵如何沿一组相互正交的方向分别缩放。这一思想将成为推导奇异值分解的基础。

## （三）奇异值分解

### 1. 从特征分解到奇异值分解

特征分解存在两个限制：

1. 它只能直接应用于方阵。
2. 一般方阵不一定可以进行完整的实数特征分解。

然而，在深度学习中，权重矩阵和梯度矩阵经常是非方阵。例如 $A\in\mathbb{R}^{m\times n}, m\neq n$。我们希望找到一种方法，对任意形状的实矩阵进行类似的方向与尺度分析。核心思路是：先将任意矩阵转换为实对称半正定矩阵，再利用正交特征分解。

对于任意实矩阵 $A\in\mathbb{R}^{m\times n}$，构造 $B=A^\top A$，其中，$A^\top\in\mathbb{R}^{n\times m}, B\in\mathbb{R}^{n\times n}$。虽然 $A$ 可能是非方阵，但 $B$ 一定是方阵。此外，$B$ 一定是实对称半正定矩阵。

性质一：对称性。
$$
\begin{equation}
B^\top=(A^\top A)^\top=A^\top A=B
\end{equation}
$$
性质二：半正定性。对于任意向量 $x\in\mathbb{R}^n$：
$$
\begin{equation}
\begin{aligned}
x^\top Bx &=x^\top A^\top Ax\\ &=(Ax)^\top(Ax)\\ &=\|Ax\|_2^2\\ &\geq 0
\end{aligned}
\end{equation}
$$
因此，$B$ 一定可以进行正交特征分解：
$$
\begin{equation}
A^\top A=V\Lambda V^\top
\end{equation}
$$
其中，$V$ 是正交矩阵，$\Lambda$ 是对角矩阵。由于 $A^\top A$ 是半正定矩阵，它的所有特征值都非 $\lambda_i\geq 0$。由此，便将任意形状矩阵 $A$ 的分析问题，转换成了一个可以进行正交特征分解的问题。

### 2. 从特征值推导奇异值

设 $v_i$ 是 $A^\top A$ 的单位特征向量，对应特征值 $\lambda_i$：
$$
\begin{equation}
A^\top Av_i=\lambda_i v_i
\end{equation}
$$
考察 $Av_i$ 的长度：
$$
\begin{equation}
\begin{aligned}
\|Av_i\|_2^2 &=(Av_i)^\top(Av_i)\\ &=v_i^\top A^\top Av_i\\ &=v_i^\top(\lambda_i v_i)\\ &=\lambda_i\|v_i\|_2^2\\ &=\lambda_i
\end{aligned}
\end{equation}
$$
因此：
$$
\begin{equation}
\|Av_i\|_2=\sqrt{\lambda_i}
\end{equation}
$$
定义：
$$
\begin{equation}
\sigma_i=\sqrt{\lambda_i}
\end{equation}
$$
其中 $\sigma_i$ 称为矩阵 $A$ 的奇异值（Singular Value）。奇异值具有明确的几何含义：矩阵 $A$ 作用于单位向量 $v_i$ 后，输出向量的长度为 $\sigma_i$。因为 $\lambda_i\geq 0$，所以奇异值始终是**非负实数**。

### 3. 左右奇异向量

对于 $\sigma_i>0$，定义：
$$
\begin{equation}
u_i=\frac{Av_i}{\sigma_i} \label{eq:defination}
\end{equation}
$$
因此有：
$$
\begin{equation}
\|u_i\|_2 =\frac{\|Av_i\|_2}{\sigma_i} =1
\end{equation}
$$
即 $u_i$ 是单位向量。接下来证明，不同的 $u_i$ 之间相互正交。对于 $\sigma_i,\sigma_j>0$：
$$
\begin{equation}
\begin{aligned}
u_i^\top u_j &=\frac{v_i^\top A^\top Av_j} {\sigma_i\sigma_j}\\ &=\frac{\lambda_j v_i^\top v_j} {\sigma_i\sigma_j}
\end{aligned}
\end{equation}
$$
由于 $V$ 是正交矩阵，有 $v_i^\top v_j=\delta_{ij}$，其中对于 $\delta_{ij}$ ：当 $i=j$ 时取 1，否则取 0。因此：
$$
\begin{equation}
u_i^\top u_j=\delta_{ij}
\end{equation}
$$
这说明，所有对应非零奇异值的 $u_i$ 两两正交。此外还可以得到如下关系：
$$
\begin{equation}
Av_i=\sigma_i u_i \label{eq:relation}
\end{equation}
$$
这就是奇异值分解的核心关系：

- $v_i$：输入空间中的右奇异向量；
- $\sigma_i$：对应方向上的缩放倍数；
- $u_i$：输出空间中的左奇异向量。

当 $\sigma_i=0$ 时，$Av_i=0$，式 $\eqref{eq:defination}$ 无法用于定义 $u_i$。此时可以在输出空间中补充适当的正交基，构成完整的 SVD。

### 4. SVD 的矩阵形式

设矩阵 $A$ 的秩为 $r$，将所有非零奇异值及其对应的奇异向量排列起来：
$$
\begin{align}
U_r&=[u_1,u_2,\ldots,u_r] \\
V_r&=[v_1,v_2,\ldots,v_r] \\
\Sigma_r &= \operatorname{diag}(\sigma_1,\sigma_2,\ldots,\sigma_r)
\end{align}
$$
其中：$U_r\in\mathbb{R}^{m\times r}, V_r\in\mathbb{R}^{n\times r}, \Sigma_r\in\mathbb{R}^{r\times r}$​。对于矩阵 $A$ 有：
$$
\begin{equation}
\begin{aligned}
A &=AI\\ &=AVV^\top\\ &=\begin{bmatrix}Av_1&Av_2&\cdots&Av_n\end{bmatrix} \begin{bmatrix} v_1^\top\\ v_2^\top\\ \vdots\\ v_n^\top \end{bmatrix}\\ &=\sum_{i=1}^{n}(Av_i)v_i^\top
\end{aligned}
\end{equation}
$$
将公式 $\eqref{eq:relation}$ 代入得到：
$$
\begin{equation}
A=\sum_{i=1}^{n}\sigma_i u_i v_i^\top
\end{equation}
$$
对于奇异值为零的项，$Av_i=0$，因此这些项全部为零，进一步得到：
$$
\begin{equation}
A=\sum_{i=1}^{r}\sigma_i u_i v_i^\top
\end{equation}
$$
写成矩阵形式有：
$$
\begin{equation}
A=U_r\Sigma_r V_r^\top
\end{equation}
$$
这就是紧凑奇异值分解（Compact SVD）。也可以补充零奇异值对应的基向量，得到完整形式：
$$
\begin{equation}
A=U\Sigma V^\top
\end{equation}
$$
其中：$U\in\mathbb{R}^{m\times m}, \Sigma\in\mathbb{R}^{m\times n}, V\in\mathbb{R}^{n\times n}$。此时 $U,V$ 都是正交矩阵，$\Sigma$ 的主对角线包含奇异值，其余元素为零，这就是奇异值分解（Singular Value Decomposition，SVD）。

### 5. SVD 的几何意义

SVD 将任意实矩阵的线性变换分解为三个步骤：

第一步：$V^\top$。将输入向量转换到右奇异向量构成的正交坐标系中。由于 $V$ 是正交矩阵，这一步不改变向量的长度。

第二步：$\Sigma$。沿着各个奇异方向分别缩放，缩放倍数由奇异值决定。例如，如果 $\Sigma= \begin{bmatrix} 10&0\\ 0&1 \end{bmatrix}$，那么两个正交方向上的缩放倍数分别是 10 和 1。

第三步：$U$。将缩放后的向量映射到输出空间对应的左奇异向量方向。由于 $U$ 是正交矩阵，这一步同样不改变向量的长度。

因此，SVD 将矩阵变换拆分成了正交变换、方向缩放和正交变换。对于任意输入向量 $x=\sum_i \alpha_i v_i$，有：
$$
\begin{equation}
Ax=\sum_i \alpha_i\sigma_i u_i
\end{equation}
$$
由于左奇异向量 $u_i$ 两两正交，因此：
$$
\begin{equation}
\|Ax\|_2^2=\sum_i\sigma_i^2\alpha_i^2
\end{equation}
$$
可以看出，奇异值越大，输入向量在对应方向上的分量对输出长度的贡献通常越大；奇异值为零时，该方向上的分量会被完全消除。

### 6. 特征分解与 SVD 的关系

特征分解与 SVD 具有相同的数学基础：都通过特殊方向和对应的缩放系数来描述矩阵变换。两者的关键区别如下：

| 比较维度 | 特征分解            | 奇异值分解                   |
| -------- | ------------------- | ---------------------------- |
| 适用对象 | 方阵                | 任意形状的实矩阵             |
| 核心向量 | 特征向量            | 左、右奇异向量               |
| 核心数值 | 特征值              | 奇异值                       |
| 数值性质 | 可以为负数或复数    | 始终为非负实数               |
| 一般形式 | $A=Q\Lambda Q^{-1}$ | $A=U\Sigma V^\top$           |
| 正交性   | 一般矩阵不保证      | 左右奇异向量均可取正交标准基 |

# 七、Muon

## （一）基本思想

AdamW 根据每个参数元素的历史梯度尺度调整步长，Muon 则把整个更新矩阵作为处理对象。它的基本思想是：先累积梯度形成动量，再对动量矩阵做近似正交化，最后更新参数。

对于矩阵参数 $W\in\mathbb{R}^{m\times n}$，先计算当前 batch 的梯度，并累积动量：

$$
\begin{equation}
\begin{aligned}
G_t&=\nabla_W L_t(W_t),\\
M_t&=\mu M_{t-1}+G_t,\quad M_0=0.
\end{aligned}
\end{equation}
$$

其中 $\mu$ 是动量系数。接下来，对动量矩阵进行正交化。结合上一节的 SVD，若：

$$
\begin{equation}
M_t=U_r\Sigma_rV_r^\top
\end{equation}
$$

其中 $r$ 是矩阵的秩，那么理想化的正交化结果是：

$$
\begin{equation}
O_t=\operatorname{Ortho}(M_t)=U_rV_r^\top
\end{equation}
$$

也就是保留左右奇异向量，将非零奇异值调整为 1。这样，原来奇异值较大的方向不再占据那么大的相对权重，较弱的非零方向也能参与更新。

最后，用正交化后的矩阵更新参数：

$$
\begin{equation}
W_{t+1}=W_t-\eta_t O_t
\end{equation}
$$

其中 $\eta_t$ 是学习率。这里正交化的是更新矩阵 $M_t$，参数矩阵 $W_t$ 本身不要求正交。实际计算时，Muon 使用 Newton–Schulz 迭代近似得到 $O_t$。

## （二）Newton–Schulz 迭代

直接通过 SVD 得到 $U_rV_r^\top$ 的成本较高。Newton–Schulz（NS）迭代用少量矩阵乘法，近似完成同样的奇异值调整。

首先归一化动量矩阵：

$$
\begin{equation}
X_0=\frac{M_t}{\|M_t\|_F+\epsilon}
\end{equation}
$$

其中 $\epsilon$ 是防止除零的数值稳定项。然后重复以下迭代：

$$
\begin{equation}
X_{k+1}
=aX_k+b(X_kX_k^\top)X_k+c(X_kX_k^\top)^2X_k
\end{equation}
$$

为什么这样的矩阵运算能实现近似正交化？设当前矩阵为 $X_k=U_r\Sigma_kV_r^\top$，代入上式可以得到：

$$
\begin{equation}
\begin{aligned}
X_{k+1}
&=aU_r\Sigma_kV_r^\top
+bU_r\Sigma_k^3V_r^\top
+cU_r\Sigma_k^5V_r^\top\\
&=U_r\left(a\Sigma_k+b\Sigma_k^3+c\Sigma_k^5\right)V_r^\top
\end{aligned}
\end{equation}
$$

因此，在精确算术下，这个过程保留左右奇异向量，只对每个非零奇异值应用同一个多项式：

$$
\begin{equation}
\sigma_{k+1}=a\sigma_k+b\sigma_k^3+c\sigma_k^5
\end{equation}
$$

通过选择合适的系数，就可以把不同大小的非零奇异值逐步调整到接近 1 的尺度，从而近似得到正交化后的更新矩阵。原始 Muon 的快速版本采用：

$$
\begin{equation}
(a,b,c)=(3.4445,-4.7750,2.0315),
\quad K=5
\end{equation}
$$

迭代 $K$ 次后，取 $O_t=X_K$ 用于参数更新。这组系数追求少量迭代内的近似效果，不保证奇异值精确收敛到 1；精确为零的奇异值仍保持为零。



# 八、总结回顾

回顾总结以上参数更新的方法：

| 优化器 | 核心做法 | 主要解决的问题 |
| --- | --- | --- |
| SGD | 使用当前 mini-batch 的梯度更新参数 | 以可承受的成本近似全量梯度下降 |
| Momentum | 累积历史梯度形成动量 | 平滑梯度波动，积累持续一致的方向 |
| Nesterov Momentum | 利用惯性预计到达位置的梯度进行修正 | 在本次更新完成前，修正惯性可能造成的过冲 |
| Adam | 一阶矩平滑方向，二阶矩调整逐元素尺度，并修正零初始化偏差 | 根据各参数的历史梯度尺度，自适应调整步长 |
| AdamW | 在 Adam 更新之外单独进行权重衰减 | 使参数衰减不受 Adam 逐元素预条件的影响 |
| Muon | 先形成动量，再近似正交化更新矩阵 | 调整矩阵更新的奇异值谱，控制不同奇异方向的相对强度 |
{% endraw %}