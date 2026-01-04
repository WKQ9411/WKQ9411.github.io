---
layout: post
title: 【论文解读】Denoising Diffusion Probabilistic Models
date: 2026-01-01 16:00:00 +08:00
summary: 解读Denoising Diffusion Probabilistic Models（DDPM）—— Diffusion模型奠基之作（内含大量推导）。
categories: Paper
---

# 目录
{:.no_toc}

* TOC
{:toc}

---

# 概览

扩散概率模型(diffusion probabilistic models)，简称扩散模型(diffusion model)，是一个马尔可夫链，包括前向过程和反向过程，前向过程是有具体的表达式可以计算的，后向过程是利用神经网络来学习的。前向过程，即扩散过程，就是不断地对图像添加高斯噪声，直到图像完全被高斯噪声淹没，如下图中的$q(\mathbf{x}_t|\mathbf{x}_{t−1})$。而反向过程，即去噪过程，就是逐渐去除噪声生成图片的过程，如下图中的$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$。

<div align="center"><img src="https://wkqpicture.oss-cn-beijing.aliyuncs.com/img/20250602100707627.png" width="80%" alt="image-20250602100700513"></div>

这里结合两篇论文来看，分别是：

1.   [Deep Unsupervised Learning using  Nonequilibrium Thermodynamics](http://arxiv.org/abs/1503.03585)：这篇论文受非平衡统计物理学启发，首先提出了扩散概率模型。

2.   [Denoising Diffusion Probabilistic Models](http://arxiv.org/abs/2006.11239)：这篇文章是将 diffusion model 用于图像生成领域的关键论文。

# 一、背景

首先给出概念：扩散模型是一种潜变量模型（latent variable model），即这类模型假设我们观察到的数据（$\mathbf{x}_0$）是由一些未观察到的、隐藏的变量（即潜变量 $\mathbf{x}_1, \dots, \mathbf{x}_T$）生成的。

下面介绍扩散模型相关背景，本部结合论文1、2一起来看。

## （一）前向过程

假设 $\mathbf{x}_0$ 是原始的、未经处理的数据（例如，一张清晰的图片），下标 0 代表时间步 $t=0$。$q(\mathbf{x}_0)$ 是初始数据的分布。通过**重复应用**一个马尔可夫扩散核 $T_\pi(\mathbf{y}|\mathbf{y}'; \beta)$，原始数据分布被逐渐转化为一个性质良好（解析上易于处理）的分布 $\pi(\mathbf{y})$，其中 $\beta$ 是扩散速率。

-   这个 $\pi(\mathbf{y})$ 就是前向过程的目标分布。通常，这个目标分布是一个非常简单的、我们熟知的分布，比如标准正态分布（高斯噪声）。“性质良好”或“解析上易于处理”意味着我们可以很容易地从这个分布中采样，或者计算它的概率密度。
-   $T_\pi(\mathbf{y}|\mathbf{y}'; \beta)$ 指的是一个转移概率函数，它定义了从一个状态 $\mathbf{y}'$ 转换到另一个状态 $\mathbf{y}$ 的概率。这个转换过程具有**马尔可夫性质**，即下一个状态 $\mathbf{y}$ 只依赖于当前状态 $\mathbf{y}'$，而与更早之前的状态无关。$T_\pi(\mathbf{y}|\mathbf{y}'; \beta)$ 这个核函数描述了单步扩散，即给定当前状态 $\mathbf{y}'$，它会以一定的概率（由 $\beta$ 控制）将其“扩散”成 $\mathbf{y}$。下标 $\pi$ 表示这个核是设计用来最终趋向于 $\pi(\mathbf{y})$ 分布的。

可以写出如下公式：

$$
\begin{align}
&\pi(\mathbf{y}) = \int T_\pi(\mathbf{y}|\mathbf{y}'; \beta) \pi(\mathbf{y}') d\mathbf{y}' \\
&q(\mathbf{x}_{t}|\mathbf{x}_{t-1}) = T_\pi(\mathbf{x}_{t}|\mathbf{x}_{t-1}; \beta_t)
\end{align}
$$

因此，从初始数据开始，执行 $T$ 次扩散的**前向过程**由下式给出：

$$
\begin{equation}
\begin{aligned}
q(\mathbf{x}_{0:T}) &= q(\mathbf{x}_{0}) q(\mathbf{x}_{1}|\mathbf{x}_{0}) q(\mathbf{x}_{2}|\mathbf{x}_{0}\mathbf{x}_{1}) \dots q(\mathbf{x}_{T}|\mathbf{x}_{0:T-1}) &(\text{概率乘法公式}) \\
& = q(\mathbf{x}_{0}) q(\mathbf{x}_{1}|\mathbf{x}_{0}) q(\mathbf{x}_{2}|\mathbf{x}_{1}) \dots q(\mathbf{x}_{T}|\mathbf{x}_{T-1}) &(\text{马尔科夫性质}) \\
&= q(\mathbf{x}_{0}) \prod_{t=1}^T q(\mathbf{x}_{t}|\mathbf{x}_{t-1})
\end{aligned}
\end{equation}
$$

其中，$q(\mathbf{x}_{t}|\mathbf{x}_{t-1})$（即单步前向转移概率）要么对应于向具有单位协方差的高斯分布进行高斯扩散，要么对应于向一个独立的二项分布进行二项扩散。前者适用于连续数据，后者适用于离散数据。

>   多元高斯分布的一般形式：
>
>   设随机向量 $\mathbf{X} = [X_1, X_2, \dots, X_n]^T$ 服从 $n$ 维高斯分布，记为 $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，其中： 
>
>   -   $\boldsymbol{\mu}$ 是均值向量（各变量的期望值）；
>   -   $\boldsymbol{\Sigma}$ 是 $n \times n$ 的协方差矩阵，对角线元素 $\Sigma_{ii} = \text{Var}(X_i)$ 为各变量的方差，非对角线元素 $\Sigma_{ij} = \text{Cov}(X_i, X_j)$ 为变量间的协方差。
>
>   单位协方差的含义：若协方差矩阵 $\boldsymbol{\Sigma} = \mathbf{I}$（单位矩阵），则每个变量的方差为1($\text{Var}(X_i) = 1$)，且任意两个变量之间的协方差为 0 ($\text{Cov}(X_i, X_j) = 0$, $i \neq j$)。此时变量间相互独立且标准化，分布称为标准多元高斯分布。

根据条件概率分布：

$$
\begin{equation}
q(\mathbf{x}_{0:T}) = q(\mathbf{x}_{0})q(\mathbf{x}_{1:T}|\mathbf{x}_{0})
\end{equation}
$$

扩散模型与其他潜变量模型的区别在于，其后验概率 $q(\mathbf{x}_{1:T}|\mathbf{x}_{0})$，即前向过程或扩散过程，被固定为一个马尔科夫链，根据方差调度 $\beta_1, \dots \beta_T$，逐次向数据中添加高斯噪声。根据前面的公式，可知：

$$
\begin{equation}
\boxed{ q(\mathbf{x}_{1:T}|\mathbf{x}_{0}) = \prod_{t=1}^T q(\mathbf{x}_{t}|\mathbf{x}_{t-1}) }
\end{equation}
$$

其单步转移概率是人为设计的，论文2中定义为：

$$
\begin{equation}
\boxed{ q(\mathbf{x}_t|\mathbf{x}_{t-1}) := \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I}) }
\end{equation}
$$

它是关于 $\mathbf{x}_t$ 的高斯分布，其均值为 $\sqrt{1-\beta_t}\mathbf{x}_{t-1}$，方差为 $\beta_t\mathbf{I}$。

## （二）反向过程

反向过程需要训练一个**生成分布 (The generative distribution)**，即模型学习到的用于生成数据的概率分布，通常用 $p_\theta$（其中 $\theta$ 代表模型参数）来表示。其目标是学习一个从纯噪声 $\mathbf{x}_{T}$ 出发，逐步去噪，最终生成数据 $\mathbf{x}_{0}$ 的过程，它同样是一个马尔科夫链：

$$
\begin{equation}
p(\mathbf{x}_{T})= \pi(\mathbf{x}_{T})
\end{equation}
$$

$$
\begin{equation}
\boxed{ p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_{T}) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t}) }
\end{equation}
$$

$p(\mathbf{x}_{T})$ 表示反向生成模型在时刻 $T$ 的分布（注意这里 $p(\mathbf{x}_{T})$ 没有下标 $\theta$，因为它通常是 $\pi(\mathbf{x}_{T})$）。$\pi(\mathbf{x}_{T})$ 是前向过程在 $T$ 步后达到的目标（通常是简单的、已知的）噪声分布，比如标准正态分布 $\mathcal{N}(\mathbf{0}, \mathbf{I})$。反向生成过程从前向过程最终到达的那个已知的噪声分布开始，这是连接前向过程和反向过程的桥梁。因此，$p(\mathbf{x}_T) = \mathcal{N}(\mathbf{x}_T; \mathbf{0}, \mathbf{I})$。$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t})$ 是反向过程的核心，表示给定时刻 $t$ 的状态 $\mathbf{x}_{t}$，生成（或去噪得到）时刻 $t-1$ 的状态 $\mathbf{x}_{t-1}$ 的条件概率，这些是模型需要学习的部分。

根据 Feller 等人的研究，对于高斯扩散和二项扩散，在连续扩散（即步长 $\beta$ 很小的极限情况）下，扩散过程的逆过程与前向过程具有**相同的函数形式**。因此，由于 $q(\mathbf{x}_{t}|\mathbf{x}_{t-1})$ 是一个高斯（或二项）分布，并且如果 $\beta_t$ 很小，那么 $q(\mathbf{x}_{t-1}|\mathbf{x}_{t})$ 也将是一个高斯（或二项）分布（注意，这里只是相同的分布，但并分布的参数也相同）。轨迹越长，扩散率 $\beta$ 就可以设置得越小。这为我们选择反向过程中 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t})$ 的函数形式提供了理论基础。

在学习过程中，对于高斯扩散核，只需要估计其均值和协方差；对于二项核，只需要估计其比特翻转概率。因此，如果反向转移 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t})$ 被建模为高斯分布，那么模型需要学习预测这个高斯分布的均值和协方差。$f_\mu(\mathbf{x}_{t}, t)$ 和 $f_\Sigma(\mathbf{x}_{t}, t)$ 是为高斯情况定义反向马尔可夫转移的均值和协方差的函数，而 $f_b(\mathbf{x}_{t}, t)$ 是为二项分布提供比特翻转概率的函数。这里引入了具体的函数 $f_\mu, f_\Sigma, f_b$ 来参数化反向过程的转移概率，通常是神经网络。它们都以当前状态 $\mathbf{x}_{t}$ 和当前时间步 $t$ 作为输入。 $f_\mu(\mathbf{x}_{t}, t)$: 预测高斯转移的均值 $\mu(\mathbf{x}_{t}, t)$。$f_\Sigma(\mathbf{x}_{t}, t)$: 预测高斯转移的协方差 $\Sigma(\mathbf{x}_{t}, t)$。$f_b(\mathbf{x}_{t}, t)$: 预测二项分布的比特翻转概率。因此，运行此算法的计算成本是这些函数的成本乘以时间步数。在论文1的所有结果中，都使用多层感知机（MLP）来定义这些函数。

在论文2中，给出反向过程的单步转移概率分布为：

$$
\begin{equation}
\boxed{ p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t) := \mathcal{N}(\mathbf{x}_{t-1}; \mu_{\theta}(\mathbf{x}_t, t), \Sigma_{\theta}(\mathbf{x}_t, t)) }
\end{equation}
$$

即转移概率分布函数为关于 $\mathbf{x}_{t-1}$ 的高斯分布，其均值 $\mu_{\theta}(\mathbf{x}_t, t)$ 和协方差 $\Sigma_{\theta}(\mathbf{x}_t, t))$ 是关于 $\mathbf{x}_t, t$ 的函数，它们是由模型学习得到的。

## （三）模型概率

生成模型赋予观测数据 $\mathbf{x}_0$的预测概率为（要计算边缘概率密度，就是对联合概率密度的其他所有随机变量求积分）：

$$
\begin{equation}
p_\theta(\mathbf{x}_{0}) = \int p_\theta(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T}
\end{equation}
$$

其中，$p_\theta(\mathbf{x}_{0:T})$ 表示整个反向（生成）轨迹 $(\mathbf{x}_{0}, \mathbf{x}_{1}, \dots, \mathbf{x}_{T})$ 的联合概率。它描述了从噪声 $\mathbf{x}_{T}$ 一路生成到数据 $\mathbf{x}_{0}$ 的完整路径的概率。$\int d\mathbf{x}_{1:T}$ 表示对所有可能的中间潜变量（即轨迹 $\mathbf{x}_{1}, \dots, \mathbf{x}_{T}$）进行积分。即，要得到模型赋予特定数据 $\mathbf{x}_{0}$ 的概率，我们需要考虑所有可能生成该 $\mathbf{x}_{0}$ 的潜变量路径 $(\mathbf{x}_{1}, \dots, \mathbf{x}_{T})$，并将这些路径的联合概率 $p_\theta(\mathbf{x}_{0:T})$ 积分起来。

直接计算这个积分通常是不可行的，但是借鉴退火重要性采样（annealed importance sampling）和 Jarzynski 等式的思想，可以转而评估前向和反向轨迹的相对概率，并在前向轨迹上取平均：

$$
\begin{align}
p_\theta(\mathbf{x}_{0}) &= \int p_\theta(\mathbf{x}_{0:T}) \frac{q(\mathbf{x}_{1:T}|\mathbf{x}_{0})}{q(\mathbf{x}_{1:T}|\mathbf{x}_{0})} d\mathbf{x}_{1:T} \\
&= \int q(\mathbf{x}_{1:T}|\mathbf{x}_{0}) \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_{0})} d\mathbf{x}_{1:T} \label{eq:expect}\\
&= \int q(\mathbf{x}_{1:T}|\mathbf{x}_{0}) \cdot p(\mathbf{x}_{T}) \prod_{t=1}^T \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t}|\mathbf{x}_{t-1})} d\mathbf{x}_{1:T}
\end{align}
$$

这里的核心思想是**重要性采样 (importance sampling)**。我们不直接从难以采样的 $p_\theta(\mathbf{x}_{0:T})$ 中采样，而是从一个更容易采样的**提议分布 (proposal distribution)** 中采样，并用一个权重来修正。在这里，前向过程 $q$ 扮演了提议分布的角色。上式首先在被积函数中乘以并除以同一个量 $q(\mathbf{x}_{1:T}|\mathbf{x}_{0})$。这个量是给定真实数据 $\mathbf{x}_{0}$ 时，前向（加噪）过程产生特定潜变量轨迹 $(\mathbf{x}_{1}, \dots, \mathbf{x}_{T})$ 的概率。现在，在公式$\eqref{eq:expect}$中，积分可以被看作是关于分布 $q(\mathbf{x}_{1:T}|\mathbf{x}_{0})$ 求期望。即：

$$
p_\theta(\mathbf{x}_{0}) = \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T}|\mathbf{x}_{0})} \left[ \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_{0})} \right]
= \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T}|\mathbf{x}_{0})} \left[ p(\mathbf{x}_{T}) \prod_{t=1}^T \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t}|\mathbf{x}_{t-1})} \right]
$$

其中 $\frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_{0})}$ 即为**重要性权重 (importance weight)**。上式可以通过从前向过程 $q(\mathbf{x}_{1:T}|\mathbf{x}_{0})$ 中抽取的样本进行平均来快速评估。这是**蒙特卡洛估计 (Monte Carlo estimation)** 的标准做法。由于直接计算这个期望（即积分）是困难的，我们可以通过以下步骤来近似它：

1.   从已知的、固定的前向过程 $q(\mathbf{x}_{1:T}|\mathbf{x}_{0})$ 中抽取多条轨迹，即潜变量序列 $(\mathbf{x}_{1}, \dots, \mathbf{x}_{T})$ 。给定一个 $\mathbf{x}_{0}$，这是可以做到的，因为前向过程是预先定义好的。
2.   对于每一条抽取的轨迹，计算重要性权重项：$W = p(\mathbf{x}_{T}) \prod_{t=1}^T \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t}|\mathbf{x}_{t-1})}$。
3.   将所有抽取样本计算得到的权重 $W$ 进行平均。这个平均值就是对 $p_\theta(\mathbf{x}_{0})$ 的一个估计。训练的过程就是通过优化神经网络，使这个估计更准确。

其中：

-   对于 $p(\mathbf{x}_{T})$：直接使用预设的简单分布（如标准高斯）来计算其在 $\mathbf{x}_{T}$ 点的概率密度。
-   对于 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t})$：将当前状态 $\mathbf{x}_{t}$ 和时间步 $t$ 输入到已训练（或正在训练）的神经网络中，得到定义该条件概率分布的参数（如高斯分布的均值和方差），然后计算 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t})$ 的值。

## （四）训练目标

训练的目标是最大化模型的对数似然。这是大多数生成模型的标准训练原则，即**最大似然估计 (Maximum Likelihood Estimation, MLE)**。我们希望调整模型的参数，使得模型赋予真实观测数据 $\mathbf{x}_0$ 的概率 $p_\theta(\mathbf{x}_{0})$ 尽可能大，也就是其对数 $\log p_\theta(\mathbf{x}_{0})$ 尽可能大。

$$
\begin{align}
L' &= \int q(\mathbf{x}_{0}) \log p_\theta(\mathbf{x}_{0}) d\mathbf{x}_{0} \\
&= \int q(\mathbf{x}_{0}) \cdot  \log \left[ \int q(\mathbf{x}_{1:T}|\mathbf{x}_{0}) \cdot p(\mathbf{x}_{T}) \prod_{t=1}^T \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t}|\mathbf{x}_{t-1})} d\mathbf{x}_{1:T} \right] d\mathbf{x}_{0}
\end{align}
$$

$L'$ 代表了希望最大化的目标函数，即在真实数据分布 $q(\mathbf{x}_{0})$ 下，模型对数似然 $\log p_\theta(\mathbf{x}_{0})$ 的**期望值**。它可以通过**琴生不等式（Jensen’s inequality）** 得到一个下界。琴生不等式：对于上凸函数 $f(x)$，比如 $\log (x)$，有 $\mathbb{E}[f(X)] \leq f(\mathbb{E}[X])$。在上述公式中，$\log$ 函数作用于一个期望（积分）之外。我们可以将 $\log$ 函数移到期望（积分）内部，从而得到一个下界：$\log \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T}|\mathbf{x}_{0})}[W] \geq \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T}|\mathbf{x}_{0})}[\log W]$ 其中 $W = p(\mathbf{x}_{T}) \prod_{t=1}^T \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t}|\mathbf{x}_{t-1})}$ 是重要性权重。因此得到：

$$
\begin{align}
L' &\geq \int q(\mathbf{x}_{0}) \cdot \left[ \int q(\mathbf{x}_{1:T}|\mathbf{x}_{0}) \cdot \log \left( p(\mathbf{x}_{T}) \prod_{t=1}^T \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t}|\mathbf{x}_{t-1})} \right) d\mathbf{x}_{1:T} \right] d\mathbf{x}_{0} \\
& = \int \left[ \int q(\mathbf{x}_{0}) \cdot q(\mathbf{x}_{1:T}|\mathbf{x}_{0}) \cdot \log \left( p(\mathbf{x}_{T}) \prod_{t=1}^T \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t}|\mathbf{x}_{t-1})} \right) d\mathbf{x}_{1:T} \right] d\mathbf{x}_{0} \\
&= \int q(\mathbf{x}_{0:T}) \cdot \log \left[ p(\mathbf{x}_{T}) \prod_{t=1}^T \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t}|\mathbf{x}_{t-1})} \right] d\mathbf{x}_{0:T} \\
&= \mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})}[\log W]
\end{align}
$$

即**下界**是就是 $\mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})}[\log W]$，**以上推导是论文1中的形式**。

**论文2中则写成了负对数似然和期望的形式**，优化目标是最小化 $L = -L'$。由于直接优化对数似然通常很困难，所以我们转而优化它的一个界限——具体来说是**证据下界 (Evidence Lower Bound, ELBO)**。最大化ELBO等价于最小化负的ELBO。具体来说，首先将论文1中的 $L'$ 添加负号，并写为期望的形式：

$$
\begin{equation}
L = -L' = -\int q(\mathbf{x}_{0}) \log p_\theta(\mathbf{x}_{0}) d\mathbf{x}_{0}
= \mathbb{E}_{\mathbf{x}_0 \sim q(\mathbf{x}_0)}[-\log p_{\theta}(\mathbf{x}_0)]
\end{equation}
$$

上面我们已经证明过，在论文1需要最大化 $L'$ 的情况下，有：

$$
\begin{equation}
L' \geq \mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})}[\log W]
\end{equation}
$$

因此，有：

$$
\begin{equation}
\begin{aligned}
L = \mathbb{E}_{\mathbf{x}_0 \sim q(\mathbf{x}_0)}[-\log p_{\theta}(\mathbf{x}_0)] 
&\leq \mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})}[-\log W] \\
&= \mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})}\left[-\log \left(p(\mathbf{x}_{T}) \prod_{t=1}^T \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{q(\mathbf{x}_{t}|\mathbf{x}_{t-1})}\right)\right] \\
&= \mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})} \left[-\log p(\mathbf{x}_T) - \sum_{t \geq 1} \log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_{t-1})}\right]
\end{aligned}
\end{equation}
$$

以上就是论文2中公式 $(3)$ 的含义。

## （五）前向过程单步转移的进一步推导

前向过程的一个显著特性是它允许以封闭形式（解析解）在任意时间步 $t$ 对 $\mathbf{x}_t$ 进行采样，这是前向过程一个非常重要的特性，它使得训练非常高效，因为我们不需要一步步迭代 $t$ 次从 $\mathbf{x}_0$ 计算得到 $\mathbf{x}_t$。令 $\alpha_t := 1 - \beta_t$ 和 $\bar{\alpha}_t := \prod_{s=1}^{t} \alpha_s$，可以得到如下公式：

$$
\begin{equation}
\boxed{ q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})} \label{eq:qx}
\end{equation}
$$


下面对上式进行推导，前文我们定义了前向过程的单步转移概率：

$$
\begin{equation}
q(\mathbf{x}_t|\mathbf{x}_{t-1}) := \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})
\end{equation}
$$

上式可以进一步写为：

$$
\begin{equation}
q(\mathbf{x}_t|\mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{\alpha_t}\mathbf{x}_{t-1}, (1-\alpha_t)\mathbf{I})
\end{equation}
$$

这意味着，$\mathbf{x}_t$ 是在均值为 $\sqrt{\alpha_t}\mathbf{x}_{t-1}$，协方差为 $(1-\alpha_t)\mathbf{I}$ 的高斯分布上采样得到的。我们可以将 $\mathbf{x}_t$ 表示为 $\mathbf{x}_{t-1}$ 的函数加上一个高斯噪声：

$$
\begin{equation}
\mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t-1}
\end{equation}
$$

其中，$\epsilon_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 是一个**标准高斯噪声**，且在不同步骤之间是**独立的**。

### 1. 迭代证明

我们从 $\mathbf{x}_0$ 开始，一步步地将 $\mathbf{x}_t$ 的表达式代入：

>   首先回顾一下，对于两个**独立的**高斯分布 $X \sim \mathcal{N}(\mu_1, \sigma_1^2)$ 和 $Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$ ，它们叠加后的分布 $aX + bY$ 满足： $aX + bY + c \sim \mathcal{N}(a\mu_1 + b\mu_2 + c, a^2 \sigma_1^2 + b^2 \sigma_2^2)$

（1）对于 $t = 1$：

$$
\mathbf{x}_1 = \sqrt{\alpha_1}\mathbf{x}_0 + \sqrt{1-\alpha_1}\epsilon_0
$$

-   由于 $\mathbf{x}_0$ 是给定的（常数），$\epsilon_0$ 是高斯分布，所以 $\mathbf{x}_1$ 也服从高斯分布。
-   其均值为 $\mathbb{E}[\mathbf{x}_1] = \sqrt{\alpha_1}\mathbf{x}_0$，其方差为 $\text{Var}(\mathbf{x}_1) = (\sqrt{1-\alpha_1})^2\text{Var}(\epsilon_0) = (1-\alpha_1)\mathbf{I}$。
-   所以，$q(\mathbf{x}_1|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_1; \sqrt{\alpha_1}\mathbf{x}_0, (1-\alpha_1)\mathbf{I})$。
-   若令 $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$，那么 $\bar{\alpha}_1 = \alpha_1$，因此，$q(\mathbf{x}_1|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_1; \sqrt{\bar{\alpha}_1}\mathbf{x}_0, (1-\bar{\alpha}_1)\mathbf{I})$。 

（2）对于 $t = 2$ ：

$$
\mathbf{x}_2 = \sqrt{\alpha_2}\mathbf{x}_1 + \sqrt{1-\alpha_2}\epsilon_1
$$

将 $\mathbf{x}_1$ 的表达式代入有：

$$
\begin{aligned}
\mathbf{x}_2 &= \sqrt{\alpha_2}(\sqrt{\alpha_1}\mathbf{x}_0 + \sqrt{1-\alpha_1}\epsilon_0) + \sqrt{1-\alpha_2}\epsilon_1 \\
&= \sqrt{\alpha_2\alpha_1}\mathbf{x}_0 + \sqrt{\alpha_2(1-\alpha_1)}\epsilon_0 + \sqrt{1-\alpha_2}\epsilon_1 \\
&= \sqrt{\bar{\alpha}_2}\mathbf{x}_0 + \sqrt{\alpha_2-\alpha_2\alpha_1}\epsilon_0 + \sqrt{1-\alpha_2}\epsilon_1
\end{aligned}
$$

-   由于 $\mathbf{x}_0$ 是给定的，$\epsilon_0$ 和 $\epsilon_1$ 是独立的标准高斯噪声。
-   $\mathbf{x}_2$ 的均值为 $\mathbb{E}[\mathbf{x}_2] = \sqrt{\alpha_2}\mathbf{x}_0$
-   $\mathbf{x}_2$ 的方差是后面两项噪声项的方差之和（因为它们独立）：

$\text{Var}(\mathbf{x}_2) = (\sqrt{\alpha_2-\alpha_2\alpha_1})^2\mathbf{I} + (\sqrt{1-\alpha_2})^2\mathbf{I}= (\alpha_2-\alpha_2\alpha_1+1-\alpha_2)\mathbf{I}= (1-\alpha_2\alpha_1)\mathbf{I} = (1-\bar{\alpha}_2)\mathbf{I}$   

-   所以，$q(\mathbf{x}_2|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_2; \sqrt{\bar{\alpha}_2}\mathbf{x}_0, (1-\bar{\alpha}_2)\mathbf{I})$。

（3）据此我们可以进行数学归纳。可以观察到，$\mathbf{x}_t$ 可以表示为 $\sqrt{\bar{\alpha}_t}\mathbf{x}_0$ 加上一个组合噪声项。这个组合噪声项是多个独立高斯噪声 $\epsilon_0, \epsilon_1, \dots, \epsilon_{t-1}$ 的线性组合。

$$
\mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t-1}
$$

假设对于 $\mathbf{x}_{t-1}$，我们有：

$$
\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon'_{t-2}
$$

其中 $\epsilon'_{t-2} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 是一个**等效**的累积噪声。 那么：

$$
\begin{aligned}
\mathbf{x}_t &= \sqrt{\alpha_t}(\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon'_{t-2}) + \sqrt{1-\alpha_t}\epsilon_{t-1} \\
&= \sqrt{\alpha_t\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{\alpha_t(1-\bar{\alpha}_{t-1})}\epsilon'_{t-2} + \sqrt{1-\alpha_t}\epsilon_{t-1}
\end{aligned}
$$

-   因为 $\bar{\alpha}_t = \alpha_t\bar{\alpha}_{t-1}$，所以均值部分是 $\sqrt{\bar{\alpha}_t}\mathbf{x}_0$。 
-   噪声部分 $\sqrt{\alpha_t(1-\bar{\alpha}_{t-1})}\epsilon'_{t-2} + \sqrt{1-\alpha_t}\epsilon_{t-1}$ 是两个独立高斯噪声的线性组合，其均值为 0，方差为： $\text{Var}(\text{noise}) = (\alpha_t(1-\bar{\alpha}_{t-1}))\mathbf{I} + (1-\alpha_t)\mathbf{I}$ $= (\alpha_t - \alpha_t\bar{\alpha}_{t-1} + 1 - \alpha_t)\mathbf{I}$ $= (1 - \alpha_t\bar{\alpha}_{t-1})\mathbf{I}$ $= (1 - \bar{\alpha}_t)\mathbf{I}$。 所以，整个噪声项可以等效为一个新的标准高斯噪声 $\epsilon'' \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 乘以 $\sqrt{1-\bar{\alpha}_t}$。 因此，$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon''$。 这就证明了 $q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$。

### 2. 直接展开证明

此外，我们也可以将 $\mathbf{x}_t$ 直接展开，一直代入到 $\mathbf{x}_0$：

$$
\begin{aligned}
\mathbf{x}_t &= \sqrt{\alpha_t \dots \alpha_1}\mathbf{x}_0 + \sqrt{\alpha_t \dots \alpha_2(1 - \alpha_1)}\epsilon_0 + \sqrt{\alpha_t \dots \alpha_3(1 - \alpha_2)}\epsilon_1 + \\ & \dots + \sqrt{\alpha_t(1 - \alpha_{t-1})}\epsilon_{t-2} + \sqrt{1 - \alpha_t}\epsilon_{t-1} \\
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \text{CombinedNoise}
\end{aligned}
$$

$\text{CombinedNoise}$ 是独立高斯变量 $\epsilon_0, \dots, \epsilon_{t-1}$ 的线性组合。其方差为各项方差之和：

$$
\begin{aligned}
\text{Var}(\text{CombinedNoise}) = [ & \alpha_t \dots \alpha_2(1 - \alpha_1) + \alpha_t \dots \alpha_3(1 - \alpha_2) + \dots + \\ 
& \alpha_t(1 - \alpha_{t-1}) + (1 - \alpha_t)]\mathbf{I}
\end{aligned}
$$

我们来计算方括号中的标量部分：

$$
\begin{aligned}
S &= (1 - \alpha_t) + \alpha_t(1 - \alpha_{t-1}) + \alpha_t\alpha_{t-1}(1 - \alpha_{t-2}) + \dots + (\alpha_t\alpha_{t-1} \dots \alpha_2)(1 - \alpha_1) \\
&= 1 - \alpha_t + \alpha_t - \alpha_t\alpha_{t-1} + \alpha_t\alpha_{t-1} - \alpha_t\alpha_{t-1}\alpha_{t-2} + \dots + \alpha_t\alpha_{t-1} \dots \alpha_2 - \alpha_t\alpha_{t-1} \dots \alpha_2\alpha_1 \\
&= S = 1 - (\alpha_t\alpha_{t-1} \dots \alpha_1) \\
&= 1 - \bar{\alpha}_t
\end{aligned}
$$

因此，$\text{CombinedNoise}$ 的方差是 $(1 - \bar{\alpha}_t)\mathbf{I}$。 这就完整地推导出了 $q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$。这个性质对于高效训练扩散模型至关重要，因为它允许我们直接从 $\mathbf{x}_0$ 采样任意时刻的 $\mathbf{x}_t$ 而无需迭代。

这个公式给出了在给定初始数据 $\mathbf{x}_0$ 的情况下，经过 $t$ 步前向加噪过程后得到的含噪数据 $\mathbf{x}_t$ 的条件概率分布。可以得出以下几点结论：

-   这是一个高斯分布，它的均值: $\sqrt{\bar{\alpha}_t}\mathbf{x}_0$。原始数据 $\mathbf{x}_0$ 被因子 $\sqrt{\bar{\alpha}_t}$ 缩放。随着时间步 $t$ 的增加，$\beta_s > 0 \implies \alpha_s < 1$，所以 $\bar{\alpha}_t$ 会减小，导致原始信号的强度逐渐衰减。
-   协方差: $(1-\bar{\alpha}_t)\mathbf{I}$。随着 $t$ 的增加，$\bar{\alpha}_t$ 减小，所以 $1-\bar{\alpha}_t$ 增大，表明噪声的方差（强度）越来越大。当 $t = T$ 且 $\bar{\alpha}_T \approx 0$ 时，均值接近 $\mathbf{0}$，协方差接近 $\mathbf{I}$，即 $\mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$，数据几乎完全变成了标准高斯噪声。
-   这个封闭形式的解至关重要：在训练时，我们可以随机采样一个数据点 $\mathbf{x}_0$ 和一个时间步 $t$，然后直接用这个公式采样得到对应的含噪样本 $\mathbf{x}_t$，而无需模拟中间的步骤。这大大提高了训练的效率。

### 3. $\beta_t$ 的设计

$\beta_t$ 通常被设计为逐渐变大，这是为什么？我们从下式来看，其中，$\alpha_t = 1 - \beta_t$。

$$
\mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t-1}
$$

-   初期（$t$ 较小）：能够温和地处理细节
    -   当 $t$ 很小的时候，$\mathbf{x}_{t-1}$ 还非常接近原始的清晰数据 $\mathbf{x}_0$，包含了大量的精细结构和细节。
    -   如果在这些早期阶段就使用很大的 $\beta_t$（即加入大量噪声），会迅速破坏这些重要的细节信息。这会使得反向过程 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 在试图从略微去噪的 $\mathbf{x}_t$ 中恢复这些细节时变得异常困难。
    -   因此，在初期使用较小的 $\beta_t$ 值，意味着每一步只对数据做微小的扰动。这样，反向过程的对应步骤也只需要学习做微小的修正，更容易学习如何去除少量噪声并保留数据的精细特征。
-   后期（$t$ 较大）：高效地趋向纯噪声
    -   当 $t$ 很大的时候，$\mathbf{x}_{t-1}$ 已经非常嘈杂，大部分原始结构已被噪声淹没，形态上更接近目标先验分布 $\mathcal{N}(\mathbf{0}, \mathbf{I})$。
    -   此时，我们的目标是在有限的 $T$ 步内，确保数据完全转化为目标先验噪声。如果此时的 $\beta_t$ 仍然非常小，那么就需要非常非常多的步数 $T$ 才能达到这个目标，这将导致训练和采样过程都极为漫长和昂贵。
    -   在后期使用较大的 $\beta_t$ 值，可以更“猛烈”地将剩余的微弱信号结构冲刷掉，从而更高效地使 $\mathbf{x}_T$ 接近纯噪声。反向过程在这些后期步骤的任务，更多的是学习如何从几乎是纯噪声的状态中塑造出数据的整体轮廓和全局结构，而不是恢复已经被破坏殆尽的细节。

## （六）目标函数的继续推导

在第(四)节中，我们得到需要优化的目标函数为：

$$
\begin{equation}
L = \mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})} \left[-\log p(\mathbf{x}_T) - \sum_{t \geq 1} \log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_{t-1})}\right] \label{eq:elbo}
\end{equation}
$$

它是原始负似然对数的一个上界（负ELBO），我们的优化目标是让这个上界尽可能小。该损失函数 $L$ 是一个期望值，优化方法是使用随机梯度下降。上述公式可以进一步改进为：

$$
\begin{equation}
\mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})} \left[ \underbrace{D_{\text{KL}}(q(\mathbf{x}_T|\mathbf{x}_0) \parallel p(\mathbf{x}_T))}_{L_T} + \sum_{t>1} \underbrace{D_{\text{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))}_{L_{t-1}} - \underbrace{\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)}_{L_0} \right]
\end{equation}
$$

公式 $\eqref{eq:elbo}$ 虽然在理论上是正确的，但在用**蒙特卡洛方法**估计时，其梯度可能具有较高的方差，这会影响训练的稳定性和效率。改进后的公式是 $L$ 的一种等价但不同形式的表达，这种形式有助于**降低梯度的方差**。

其推导过程如下：

$$
\begin{align}
L &= \mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})} \left[-\log p(\mathbf{x}_T) - \sum_{t \geq 1} \log \frac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_{t-1})}\right] \\
&= \mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})} \left[ -\log p(\mathbf{x}_T) - \sum_{t > 1} \log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_{t-1})} - \log \frac{p_\theta(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_1|\mathbf{x}_0)} \right] \label{eq:td1} \\
&= \mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})} \left[ -\log p(\mathbf{x}_T) - \sum_{t > 1} \log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)} \cdot \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)} - \log \frac{p_\theta(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_1|\mathbf{x}_0)} \right] \label{eq:td2} \\
&= \mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})} \left[ -\log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_0)} - \sum_{t > 1} \log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)} - \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right] \label{eq:td3} \\
&= \mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})} \left[ D_{\text{KL}}(q(\mathbf{x}_T|\mathbf{x}_0) \parallel p(\mathbf{x}_T)) + \sum_{t>1} D_{\text{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)) - \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right] \label{eq:td4}
\end{align}
$$

其中：

### 1. 分步推导1

上式 $\eqref{eq:td1}$ 到 $\eqref{eq:td2}$ 的推导原理如下：

针对 $\eqref{eq:td1}$ 中 $t > 1$ 的项 $\log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_{t-1})}$，使用 **贝叶斯定理** 和 **前向过程的马尔可夫性质** 进行重写。关键点在于引入**真实后验分布** $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$（这是在已知 $\mathbf{x}_0$ 和 $\mathbf{x}_t$ 下 $\mathbf{x}_{t-1}$ 的条件分布）。

-   由前向扩散的马尔可夫性，有：

$$
q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0) = q(\mathbf{x}_t|\mathbf{x}_{t-1}) \quad (\text{因为前向过程只依赖前一状态})
$$

-   应用贝叶斯定理于后验分布：

$$
q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \frac{q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0)q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)} = \frac{q(\mathbf{x}_t|\mathbf{x}_{t-1})q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)}
$$

-   由此解出 $q(\mathbf{x}_t|\mathbf{x}_{t-1})$：

$$
q(\mathbf{x}_t|\mathbf{x}_{t-1}) = q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)\frac{q(\mathbf{x}_t|\mathbf{x}_0)}{q(\mathbf{x}_{t-1}|\mathbf{x}_0)}
$$

-   将上式代入，就能得到式 $\eqref{eq:td2}$。

### 2. 分步推导2

上式 $\eqref{eq:td2}$ 到 $\eqref{eq:td3}$ 的推导原理如下：

-   首先展开 $\eqref{eq:td2}$ 中的 $\sum_{t > 1} \log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)} \cdot \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)}$，得到：

$$
L = \mathbb{E} \left[ -\log p(\mathbf{x}_T) - \sum_{t > 1} \log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)} - \sum_{t > 1} \log \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)} - \log \frac{p_\theta(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_1|\mathbf{x}_0)} \right]
$$

- 注意到 $\sum_{t > 1} \log \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)}$ 是 **望远镜求和（Telescoping Sum）**，可以中间项相消，只剩边界项：

  $$
  \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)} = \sum_{t=2}^T \left( \log q(\mathbf{x}_{t-1}|\mathbf{x}_0) - \log q(\mathbf{x}_t|\mathbf{x}_0) \right) = \log q(\mathbf{x}_1|\mathbf{x}_0) - \log q(\mathbf{x}_T|\mathbf{x}_0)
  $$

- 代入并重组合并：

  $$
  \begin{aligned}
  L &= \mathbb{E} \left[ -\log p(\mathbf{x}_T) - \sum_{t > 1} \log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)} - \log q(\mathbf{x}_1|\mathbf{x}_0) + \log q(\mathbf{x}_T|\mathbf{x}_0) - \log \frac{p_\theta(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_1|\mathbf{x}_0)} \right] \\
  &= \mathbb{E} \left[ \left[ -\log p(\mathbf{x}_T) + \log q(\mathbf{x}_T|\mathbf{x}_0) \right] - \sum_{t > 1} \log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)} - \left[ \log q(\mathbf{x}_1|\mathbf{x}_0) + \log \frac{p_\theta(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_1|\mathbf{x}_0)} \right] \right] \\
  &= \mathbb{E} \left[ -\log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_0)} - \sum_{t > 1} \log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)} - \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right]
  \end{aligned}
  $$

### 3. KL散度定义及其与交叉熵、信息熵的关系

由于上述推导最终转换成了KL散度的形式，这里首先重点了解一下KL散度的定义。

**KL散度 (Kullback-Leibler Divergence)**，也称为相对熵 (Relative Entropy)，是一种衡量两个概率分布之间差异的非对称性度量。简单来说，它量化了当我们用一个近似的概率分布 $Q$ 来描述一个真实的概率分布 $P$ 时，会损失多少信息，或者说需要多少额外的“信息量”才能表达 $P$。

给定两个概率分布 $P(x)$ 和 $Q(x)$（对于同一个随机变量 $x$）：

-   对于离散随机变量：

$$
D_{KL}(P||Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}
$$

-   对于连续随机变量：

$$
D_{KL}(P||Q) = \int_{-\infty}^{\infty} P(x) \log \frac{P(x)}{Q(x)} dx
$$

其中，$P(x)$ 通常被认为是真实的、目标的数据分布。$Q(x)$ 通常是我们用来近似 $P(x)$ 的模型或理论分布。

KL散度具有以下特性：

-   衡量差异性：KL散度衡量的是用分布 $Q$ 来近似分布 $P$ 时的“距离”或“差异程度”。值越大，差异越大。
-   非负性：$D_{KL}(P||Q) \geq 0$。
-   当且仅当 $P = Q$ 时为零：$D_{KL}(P||Q) = 0$ 当且仅当 $P(x) = Q(x)$ 对于几乎所有的 $x$ 都成立。
-   不对称性：$D_{KL}(P||Q) \neq D_{KL}(Q||P)$。

KL散度和信息熵、交叉熵有密切的关系。

**信息熵 (Entropy)** $H(P)$ 是对概率分布 $P$ 不确定性的一种度量。它表示从分布 $P$ 中随机抽取一个样本，我们平均需要多少信息量来描述这个样本。熵越大，分布的不确定性越大。

-   对于离散分布的 $P(x)$：

$$
H(P) = -\sum_{x \in \mathcal{X}} P(x) \log P(x)
$$

-   对于连续分布，也称为微分熵，则是：

$$
H(P) = -\int P(x) \log P(x) dx
$$

**交叉熵 (Cross-Entropy)** $H(P, Q)$ 衡量的是，当我们使用一个基于概率分布 $Q$ 的编码方案去编码来自真实概率分布 $P$ 的事件时，平均所需要的信息量。

-   对于离散分布 $P(x)$ 和 $Q(x)$：

$$
H(P, Q) = -\sum_{x \in \mathcal{X}} P(x) \log Q(x)
$$

-   对于连续分布，则是：

$$
H(P, Q) = -\int P(x) \log Q(x) dx
$$

以离散情况为例，我们从KL散度的定义开始：

$$
D_{KL}(P||Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}
$$

利用对数的性质 $\log(a/b) = \log a - \log b$:

$$
D_{KL}(P||Q) = \sum_{x \in \mathcal{X}} P(x)(\log P(x) - \log Q(x))
$$

展开求和：

$$
D_{KL}(P||Q) = \sum_{x \in \mathcal{X}} P(x) \log P(x) - \sum_{x \in \mathcal{X}} P(x) \log Q(x)
$$

我们认出这两个部分：

-   第一部分 $\sum_{x \in \mathcal{X}} P(x) \log P(x)$ 等于 $-H(P)$（即真实分布 $P$ 的负熵）。
-   第二部分 $-\sum_{x \in \mathcal{X}} P(x) \log Q(x)$ 等于 $H(P, Q)$（即 $P$ 和 $Q$ 之间的交叉熵）。
-   所以，我们可以得到：

$$
D_{KL}(P||Q) = H(P, Q) - H(P)
$$

这个关系式 $D_{KL}(P||Q) = H(P, Q) - H(P)$ 非常有启发性：

-   $H(P)$ (信息熵)：这是用最优编码（即基于真实分布 $P$ 的编码）来编码来自 $P$ 的事件时，平均所需要的最小信息量。
-   $H(P, Q)$ (交叉熵)： 这是用一个可能不完美的编码（即基于分布 $Q$ 的编码）来编码来自 $P$ 的事件时，平均所需要的信息量。
-   $D_{KL}(P||Q)$ (KL散度)：因此，KL散度就是这两者之差。它表示由于使用了次优的分布 $Q$ 而不是最优的分布 $P$ 来进行编码，所导致的额外平均信息量或编码长度的增加量。 换句话说，交叉熵 $H(P, Q)$ 总是大于或等于熵 $H(P)$（因为 $D_{KL}(P||Q) \geq 0$）。KL散度量化了这种“大于”的程度。

### 4. 分步推导3

下面，我们回到 $\eqref{eq:td3}$ 到 $\eqref{eq:td4}$ 的推导：

重新写出式 $\eqref{eq:td3}$ 推导至 $\eqref{eq:td4}$ 的结果如下：

$$
\begin{aligned}
&\mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})} \left[ -\log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_0)} - \sum_{t > 1} \log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)} - \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right] \\
&= \mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})} \left[ D_{\text{KL}}(q(\mathbf{x}_T|\mathbf{x}_0) \parallel p(\mathbf{x}_T)) + \sum_{t>1} D_{\text{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)) - \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right]
\end{aligned}
$$

对于 KL 散度，可以写成如下形式：$D_{\text{KL}}(P \parallel Q) = \mathbb{E}_{P} \left[ \log \frac{P}{Q} \right]$。

因此，对于 $\eqref{eq:td3}$：

- 第一项 $-\log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_0)} = \log \frac{q(\mathbf{x}_T|\mathbf{x}_0)}{p(\mathbf{x}_T)}$。在期望 $\mathbb{E}_{q(\mathbf{x}_{0:T})}$ 下（也就是 $\mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})}$），给定 $\mathbf{x}_0$，这等价于：

$$
\begin{aligned}
\mathbb{E}_{q(\mathbf{x}_{0:T})} \left[ \log \frac{q(\mathbf{x}_T|\mathbf{x}_0)}{p(\mathbf{x}_T)} \right] &= \mathbb{E}_{q(\mathbf{x}_0)} \left[ \mathbb{E}_{q(\mathbf{x}_T|\mathbf{x}_0)} \left[ \log \frac{q(\mathbf{x}_T|\mathbf{x}_0)}{p(\mathbf{x}_T)} \right] \right] \\
&= \mathbb{E}_{q(\mathbf{x}_0)} \left[ D_{\text{KL}}(q(\mathbf{x}_T|\mathbf{x}_0) \parallel p(\mathbf{x}_T)) \right]
\end{aligned}
$$

- 第二项求和项 $-\sum_{t>1} \log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)} = \sum_{t>1} \log \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}$。在期望下，给定 $\mathbf{x}_t$ 和 $\mathbf{x}_0$，这等价于：

$$
\begin{aligned}
\mathbb{E}_{q(\mathbf{x}_{0:T})} \left[ \log \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)} \right] &= \mathbb{E}_{q(\mathbf{x}_0, \mathbf{x}_t)} \left[ \mathbb{E}_{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)} \left[ \log \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)} \right] \right] \\
&= \mathbb{E}_{q(\mathbf{x}_0, \mathbf{x}_t)} \left[ D_{\text{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)) \right]
\end{aligned}
$$

- 最后一项 $-\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)$ 保持不变，它是 $\mathbf{x}_1$ 到 $\mathbf{x}_0$ 的重构损失（Reconstruction Loss），通常用均方误差等近似。

因此，实际推导结果为：

$$
\mathbb{E}_{q(\mathbf{x}_0)} \left[ D_{\text{KL}}(q(\mathbf{x}_T|\mathbf{x}_0) \parallel p(\mathbf{x}_T)) \right] + \mathbb{E}_{q(\mathbf{x}_0, \mathbf{x}_t)} \left[ D_{\text{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)) \right] -\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)
$$

由于期望的**无关变量可以边缘化**的特性，上式可以统一为最终的 $\eqref{eq:td4}$：

$$
\mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})} \left[ D_{\text{KL}}(q(\mathbf{x}_T|\mathbf{x}_0) \parallel p(\mathbf{x}_T)) + \sum_{t>1} D_{\text{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)) - \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right]
$$

（原文中将 $\mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})}$ 简写为了 $\mathbb{E}_q$）

### 5. 理解目标函数的最终推导形式

经过前文的推导，我们再次写出最终目标函数的形式：

$$
\begin{equation}
\boxed{\mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})} \left[ \underbrace{D_{\text{KL}}(q(\mathbf{x}_T|\mathbf{x}_0) \parallel p(\mathbf{x}_T))}_{L_T} + \sum_{t>1} \underbrace{D_{\text{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))}_{L_{t-1}} - \underbrace{\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)}_{L_0} \right] } \label{eq:l}
\end{equation}
$$

🌟各项含义如下：

第一项：$L_T = D_{KL}(q(\mathbf{x}_T|\mathbf{x}_0) || p(\mathbf{x}_T))$ (先验匹配项)：

-   $q(\mathbf{x}_T|\mathbf{x}_0)$：是从 $\mathbf{x}_0$ 开始，经过 $T$ 步前向加噪后得到的 $\mathbf{x}_T$ 的分布。根据公式 $\eqref{eq:qx}$，这是一个均值为 $\sqrt{\bar{\alpha}_T}\mathbf{x}_0$，协方差为 $(1-\bar{\alpha}_T)\mathbf{I}$ 的高斯分布。
-   $p(\mathbf{x}_T)$：是模型为 $\mathbf{x}_T$ 设定的先验分布，通常取为标准正态分布 $\mathcal{N}(\mathbf{0}, \mathbf{I})$。
-   含义：这一项鼓励前向过程最终产生的噪声分布 $q(\mathbf{x}_T|\mathbf{x}_0)$ 与我们设定的简单先验 $p(\mathbf{x}_T)$ 相匹配。如果 $T$ 足够大使得 $\bar{\alpha}_T \approx 0$，那么 $q(\mathbf{x}_T|\mathbf{x}_0) \approx \mathcal{N}(\mathbf{0}, (1)\mathbf{I})$，此时如果 $p(\mathbf{x}_T) = \mathcal{N}(\mathbf{0}, \mathbf{I})$，则 $L_T \approx 0$。

第二项：$L_{t-1} = \sum_{t>1} D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) || p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))$ (去噪/匹配项)：

-   这个求和从 $t=2$ 到 $T$（因为 $t > 1$）。(注意：这里 $L_{t-1}$ 的下标 $t-1$ 表示求和中每一项都是关于 $\mathbf{x}_{t-1}$ 的条件概率的)。
-   $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$：这是**真实的反向**条件概率。即给定当前噪声状态 $\mathbf{x}_t$ 和原始数据 $\mathbf{x}_0$，上一步状态 $\mathbf{x}_{t-1}$ 的真实**后验分布**。
-   $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$：这是我们模型**学习到的反向**条件概率。
-   含义：这些KL散度项是训练的**核心**。它们迫使模型学习到的反向转移 $p_\theta$ 去逼近（或“匹配”）由前向过程定义的理想反向转移 $q$。最小化这些KL散度就是在学习如何有效地“去噪”一步。

第三项：$L_0 = -\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)$ (重构项)：

-   这是在给定第一个含噪状态 $\mathbf{x}_1$ 的情况下，模型生成（或“重构”）原始数据 $\mathbf{x}_0$ 的负对数似然。
-   含义：这一项鼓励模型能够从轻微噪声的 $\mathbf{x}_1$ 中准确地恢复出原始数据 $\mathbf{x}_0$。

这一目标函数使用KL散度将 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 与前向过程的**后验概率分布**进行直接比较，这些后验分布在以 $\mathbf{x}_0$ 为条件时是易于处理的：

$$
\begin{equation}
\boxed{ q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t\mathbf{I}) }
\end{equation}
$$

其中：

$$
\begin{equation}
\boxed{
\begin{aligned}
\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) &= \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t \\
\tilde{\beta}_t &= \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
\end{aligned}}
\end{equation}
$$

上述公式推导过程如下：

首先，如前文所述，根据马尔科夫性质和贝叶斯定理，对于前向过程的后验概率分布，有：

$$
q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \frac{q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0)q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)} = \frac{q(\mathbf{x}_t|\mathbf{x}_{t-1})q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)}
$$

此外，我们已知：

$$
\begin{aligned}
q(\mathbf{x}_t|\mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I}) \\
q(\mathbf{x}_t|\mathbf{x}_{t-1}) &= \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I}) \\
\alpha_t &= 1 - \beta_t \\
\bar{\alpha}_t &= \prod_{s=1}^{t} \alpha_s
\end{aligned}
$$

根据高斯分布的性质，**高斯分布的乘积仍然是高斯分布**，因此 $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$ 也是一个高斯分布。我们知道多维高斯分布的形式为：

$$
p(\mathbf{x} \mid \mu, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (\mathbf{x} - \mu)^\top \Sigma^{-1} (\mathbf{x} - \mu) \right)
$$

其中，$\mu$ 是均值向量，$\Sigma$ 是协方差矩阵。高斯分布的指数部分是变量的二次型，因此，我们只需关注指数部分中涉及到变量 $\mathbf{x}_{t-1}$ 的部分，只要将关于 $\mathbf{x}_{t-1}$ 的部分配成二次型，我们就可以确定 $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$ 的均值和方差。

因此，$q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$ 这一概率分布满足：

$$
q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \propto q(\mathbf{x}_t | \mathbf{x}_{t-1}) \cdot q(\mathbf{x}_{t-1} | \mathbf{x}_0)
$$

代入高斯概率密度，有：

$$
\begin{aligned}
q(\mathbf{x}_t | \mathbf{x}_{t-1}) &= \mathcal{N}(\mathbf{x}_t; \sqrt{\alpha_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}) \propto \exp\left(-\frac{\|\mathbf{x}_t - \sqrt{\alpha_t} \mathbf{x}_{t-1}\|^2}{2\beta_t}\right) \\
q(\mathbf{x}_{t-1} | \mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0, (1 - \bar{\alpha}_{t-1}) \mathbf{I}) \propto \exp\left(-\frac{\|\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0\|^2}{2(1 - \bar{\alpha}_{t-1})}\right)
\end{aligned}
$$

因此，$q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$ 的**对数**密度为：

$$
\log q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \propto -\frac{1}{2} \left[ \frac{\|\mathbf{x}_t - \sqrt{\alpha_t} \mathbf{x}_{t-1}\|^2}{\beta_t} + \frac{\|\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0\|^2}{1 - \bar{\alpha}_{t-1}} \right] + C
$$

其中 $C$ 是与 $\mathbf{x}_{t-1}$ 无关的常数项。定义二次形式：

$$
L = \frac{1}{2} \left[ \frac{\|\mathbf{x}_t - \sqrt{\alpha_t} \mathbf{x}_{t-1}\|^2}{\beta_t} + \frac{\|\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0\|^2}{1 - \bar{\alpha}_{t-1}} \right]
$$

则 $q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \propto \exp(-L)$。由于这是关于 $\mathbf{x}_{t-1}$ 的二次函数，后验条件概率又是高斯分布，因此其均值和方差可通过匹配二次形式的系数得到。（即，之所只需匹配一次二次项系数就可以，是因为我们已经通过高斯分布乘法的性质知道了该分布是一个高斯分布，因此其他项无需再额外计算）

令 $\mathbf{v} = \mathbf{x}_{t-1}$，展开 $L$ 有：

$$
L = \frac{1}{2} \left[ \frac{\|\mathbf{x}_t\|^2 - 2\sqrt{\alpha_t} \mathbf{x}_t^\top \mathbf{v} + \alpha_t \|\mathbf{v}\|^2}{\beta_t} + \frac{\|\mathbf{v}\|^2 - 2\sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0^\top \mathbf{v} + \bar{\alpha}_{t-1} \|\mathbf{x}_0\|^2}{1 - \bar{\alpha}_{t-1}} \right]
$$

忽略与 $\mathbf{v}$ 无关的常数项，保留二次项和一次项：

$$
L = \frac{1}{2} \left[ \mathbf{v}^\top \mathbf{v} \left( \frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}} \right) + \mathbf{v}^\top \left( -\frac{2\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t - \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0 \right) \right] + \text{const}
$$

高斯分布的标准形式为：

$$
\mathcal{N}(\mathbf{v}; \mu, \Sigma) \propto \exp\left( -\frac{1}{2} (\mathbf{v} - \mu)^\top \Sigma^{-1} (\mathbf{v} - \mu) \right) = \exp\left( -\frac{1}{2} \mathbf{v}^\top \Sigma^{-1} \mathbf{v} + \mu^\top \Sigma^{-1} \mathbf{v} + \text{const} \right)
$$

比较系数：

- **二次项系数**：$\frac{1}{2} \mathbf{v}^\top \Sigma^{-1} \mathbf{v}$ 对应 $\frac{1}{2} \mathbf{v}^\top \mathbf{v} \left( \frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}} \right)$，因此：

  $$
  \Sigma^{-1} = \left( \frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}} \right) \mathbf{I}
  $$

- **一次项系数**：在标准形式中为 $\mu^\top \Sigma^{-1} \mathbf{v}$，在 $L$ 中为 $\mathbf{v}^\top \left( -\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t - \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0 \right)$，因此：

  $$
  \mu^\top \Sigma^{-1} = \frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t^\top + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0^\top
  $$

  两边同时转置，又因为协方差矩阵是对称矩阵，有：

  $$
  \Sigma^{-1} \mu = \frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0
  $$

可知**方差** $\tilde{\beta}_t \mathbf{I} = \Sigma$：

$$
\Sigma = \left( \frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}} \right)^{-1} \mathbf{I}
$$

计算分母：

$$
D = \frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}} = \frac{\alpha_t (1 - \bar{\alpha}_{t-1}) + \beta_t}{\beta_t (1 - \bar{\alpha}_{t-1})}
$$

代入 $\beta_t = 1 - \alpha_t$：

$$
\alpha_t (1 - \bar{\alpha}_{t-1}) + \beta_t = \alpha_t - \alpha_t \bar{\alpha}_{t-1} + 1 - \alpha_t = 1 - \alpha_t \bar{\alpha}_{t-1}
$$

由于 $\bar{\alpha}_t = \alpha_t \bar{\alpha}_{t-1}$：

$$
D = \frac{1 - \bar{\alpha}_t}{\beta_t (1 - \bar{\alpha}_{t-1})}
$$

因此：

$$
\Sigma = \frac{\beta_t (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{I} \implies \boxed{ \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t}
$$

**均值** $\tilde{\mu}_t$：

$$
\mu = \Sigma \left( \frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0 \right)
$$

代入 $\Sigma = \frac{\beta_t (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{I}$，得到：

$$
\mu = \frac{\beta_t (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \left( \frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0 \right)
$$

化简有：

$$
\begin{aligned}
\mu &= \frac{\beta_t (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \cdot \frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\beta_t (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \cdot \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0 \\
&= \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t
\end{aligned}
$$

这给出了**真实反向条件分布** $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$ 的均值 $\tilde{\mu}_t$ 和方差系数 $\tilde{\beta}_t$ 的精确解析表达式。

-   $\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0)$：均值是 $\mathbf{x}_0$ 和 $\mathbf{x}_t$ 的一个线性组合，系数由前向过程的参数 $\alpha_t, \beta_t, \bar{\alpha}_t$ 决定。
-   $\tilde{\beta}_t$：方差系数是前向过程方差 $\beta_t$ 的一个加权版本。一个非常重要的特性是，这个方差 $\tilde{\beta}_t\mathbf{I}$ 不依赖于 $\mathbf{x}_t$ 或 $\mathbf{x}_0$，它只依赖于时间步 $t$（通过 $\alpha_t, \beta_t, \bar{\alpha}_t$ 等预设参数）。这使得模型 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ **通常可以将方差设定为这些已知的** $\tilde{\beta}_t\mathbf{I}$，而只需要学习均值 $\mu_\theta(\mathbf{x}_t, t)$，这简化了学习任务。

综上，公式 $\eqref{eq:l}$ 中的所有KL散度都是高斯分布之间的比较：

-   $L_T = D_{KL}(q(\mathbf{x}_T|\mathbf{x}_0) || p(\mathbf{x}_T))$：$q(\mathbf{x}_T|\mathbf{x}_0)$ 是高斯分布，$p(\mathbf{x}_T)$ 通常被选为标准高斯分布。
-   $L_{t-1}$ 中的 $D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) || p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))$：$q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$ 是高斯分布，而 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 通常也被参数化为高斯分布（其均值由神经网络预测，方差通常固定为 $\tilde{\beta}_t\mathbf{I}$ 或其他预设值）。

两个高斯分布之间的 KL 散度具有已知的解析公式。这意味着我们可以精确地计算这些 KL 散度项，而不需要通过采样来估计它们（采样估计通常会有较高的方差）。Rao-Blackwell 定理是一种统计学中用于改进估计量的方法，通常通过将估计量的一部分替换为其条件期望来降低方差。在这里，使用 KL 散度的解析形式而不是采样估计，可以看作是利用了这个思想，因为它用一个精确的计算代替了一个可能有噪声的估计，从而降低了整体损失函数估计的方差。

# 二、扩散模型和去噪自编码器

扩散模型表面上可能看起来是一类受限的潜变量模型，但它们在实现上允许大量的自由度。 尽管前向过程固定，但研究者在设计模型时仍有很大的选择空间，包括：

1.   前向过程的方差 $\beta_t$：如何设计噪声计划表。
2.   模型架构：反向过程中使用的神经网络（如U-Net）的结构。
3.   高斯分布的参数化：反向过程中，如何从网络输出中得到高斯分布的均值和方差。

文章建立了一个在扩散模型和**去噪分数匹配 (denoising score matching)** 之间新的、明确的联系，从而将前面推导出的目标函数简化成一个更简单、更直观、带权重的形式。

## （一）前向过程与 $L_T$

我们忽略了前向过程方差 $\beta_t$ 可以通过重参数化来学习这一事实，而是将它们固定为常数。这是一个关键的设计决策。虽然理论上可以让模型自己学习最佳的噪声计划表 $\beta_t$，但作者为了简化模型，选择将 $\beta_t$ 设定为一组预先定义好的、固定的超参数。DDPM论文中使用的就是从 $10^{-4}$ 线性增加到 $0.02$ 的调度方案。

我们回顾一下 $L_T$ 的定义：

$$
L_T = D_{KL}(q(\mathbf{x}_T|\mathbf{x}_0) || p(\mathbf{x}_T))
$$

在论文的实现中，因为 $\beta_t$ 被固定了，所以整个前向加噪过程就完全确定了，不涉及任何需要通过训练来优化的参数，所以 $L_T$ 在训练过程中是一个常数，可以被忽略。

## （二）反向过程与 $L_{1:T-1}$

回顾一下 $L_{1:T-1}$，即 $L_1$ 到 $L_{T-1}$ 的和：

$$
L_{1:T-1} = \sum_{t>1} \underbrace{D_{\text{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))}_{L_{t-1}}
$$

这一过程的目的是为了使模型 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 逼近**真实的反向条件概率** $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$，后者的表达式我们上文已推导过，即：

$$
q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t\mathbf{I})
$$

现在讨论在 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t)) (1 < t \leq T)$ 中我们的选择。首先，将 $\Sigma_\theta(\mathbf{x}_t, t)$ 设定为不参与训练的、只与时间相关的常数 $\sigma_t^2\mathbf{I}$。因此模型分布为：

$$
\boxed{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \sigma_t^2\mathbf{I})}
$$

这是一个重大的简化决策，反向过程的协方差矩阵 $\Sigma_\theta$ 将不被神经网络学习，并且其方差 $\sigma_t^2$ 只依赖于时间步 $t$，而不依赖于当前的含噪数据 $\mathbf{x}_t$。如前文所述：模型 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ **通常可以将方差设定为这些已知的** $\tilde{\beta}_t\mathbf{I}$，而只需要学习均值 $\mu_\theta(\mathbf{x}_t, t)$，这简化了学习任务。

但是实验发现：

-   $\sigma_t^2 = \beta_t$：直接使用前向过程的方差。
-   $\sigma_t^2 = \tilde{\beta}_t$：使用我们之前推导出的“真实”反向条件概率 $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$ 的方差。

第一种选择对于 $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 的情况是最优的，而第二种选择对于 $\mathbf{x}_0$ 被**确定性**地设为一个点的情况是最优的。最终，作者发现这两种选择在实验中效果差不多，并在论文中选择了更简单的前者，即 $\sigma_t^2 = \beta_t$。

其次，当 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \sigma_t^2\mathbf{I})$ 时，为了表示均值 $\mu_\theta(\mathbf{x}_t, t)$，可以写出如下公式：

$$
\begin{equation}
L_{t-1} = \mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})}\left[\frac{1}{2\sigma_t^2}\|\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) - \mu_\theta(\mathbf{x}_t, t)\|^2\right] + C \label{eq:lt-1}
\end{equation}
$$

这是因为，对于两个多维高斯分布 $P_1 = \mathcal{N}(\mu_1, \Sigma_1)$ 和 $P_2 = \mathcal{N}(\mu_2, \Sigma_2)$，根据高斯分布和 KL 散度公式，可以得到它们之间的 KL 散度的解析解：

$$
D_{KL}(P_1||P_2) = \frac{1}{2}\left(\log\frac{|\Sigma_2|}{|\Sigma_1|} - D + \text{Tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2 - \mu_1)^T\Sigma_2^{-1}(\mu_2 - \mu_1)\right)
$$

其中 $D$ 是数据的维度。将我们的具体分布代入上式，即：

-   $\mu_1 = \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0)$  
-   $\mu_2 = \mu_\theta(\mathbf{x}_t, t)$  
-   $\Sigma_1 = \tilde{\beta}_t\mathbf{I}$  
-   $\Sigma_2 = \sigma_t^2\mathbf{I}$

化简后就可以得到 $L_{t-1}$ 的表达式，这里我们不再详细推导。可以看出，最小化 $L_{t-1}$ 就等价于最小化学习到的均值 $\mu_\theta$ 和目标均值 $\tilde{\mu}_t$ 之间的**加权平方距离**。$C$ 是一个不依赖于模型参数 $\theta$ 的常数项，在优化时可以忽略。

看到这里，一个直接的想法是：让神经网络直接预测目标均值 $\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0)$。但DDPM的作者提出了一个更巧妙的方法。

首先，将公式：

$$
q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})
$$

重参数化为：

$$
①:\mathbf{x}_t(\mathbf{x}_0, \epsilon) = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon
$$

其中，$\epsilon \sim \mathcal{N}(\mathbf{0},\mathbf{I})$ ，这样，$\mathbf{x}_t$ 可以表示为 $\mathbf{x}_0$ 和标准高斯噪声 $\epsilon$ 的确定性函数。由上式可以得到：

$$
②:\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\epsilon)
$$

此外，我们之前已经推导过：

$$
③:\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t
$$

将 $①\sim③$ 代入 $\eqref{eq:lt-1}$ 中，利用 $\bar{\alpha}_t = \alpha_t\bar{\alpha}_{t-1}$ 和 $\beta_t = 1 - \alpha_t$ ，化简后有：    

$$
\begin{align}
L_{t-1} - C &= \mathbb{E}_{\mathbf{x}_0, \epsilon}\left[\frac{1}{2\sigma_t^2}\left\|\tilde{\mu}_t\left(\mathbf{x}_t(\mathbf{x}_0, \epsilon), \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t(\mathbf{x}_0, \epsilon) - \sqrt{1 - \bar{\alpha}_t}\epsilon)\right) - \mu_\theta(\mathbf{x}_t(\mathbf{x}_0, \epsilon), t)\right\|^2\right] \\
&= \boxed{ \mathbb{E}_{\mathbf{x}_0, \epsilon}\left[\frac{1}{2\sigma_t^2}\left\|\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t(\mathbf{x}_0, \epsilon) - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon\right) - \mu_\theta(\mathbf{x}_t(\mathbf{x}_0, \epsilon), t)\right\|^2\right]} \label{eq:l-c}
\end{align}
$$

这里因为将所有 $\mathbf{x}_t$ 重参数为 $①$ 的形式，即 $\mathbf{x}_t(\mathbf{x}_0, \epsilon)$，每个 $\mathbf{x}_t$ 都可由 $(\mathbf{x}_0, \epsilon)$ 表达，因此期望的下标可分解为：

$$
\mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})}[\cdot] = \mathbb{E}_{\mathbf{x}_0 \sim q(\mathbf{x}_0), \epsilon \sim \mathcal{N}(0, \mathbf{I})}[\cdot]
$$

故上式将期望简写为 $\mathbb{E}_{\mathbf{x}_0, \epsilon}$。

公式 $\eqref{eq:l-c}$ 表明，在给定 $\mathbf{x}_t$ 的情况下，$\mu_\theta$ 必须预测 $\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon\right)$，这是 $\tilde{\mu}_t$ 的一种新形式，在这种形式的表达式中，$\mathbf{x}_t$ 是模型的输入，唯一不确定的是噪声 $\epsilon$，这启发我们只需让神经网络来预测这个噪声即可。我们可以选择如下的参数化方式：

$$
\begin{equation}
\boxed{ \mu_\theta(\mathbf{x}_t, t) = \tilde{\mu}_t\left(\mathbf{x}_t, \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_\theta(\mathbf{x}_t,t))\right) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_\theta(\mathbf{x}_t, t)\right) } \label{eq:mu}
\end{equation}
$$

这是模型参数化的核心。我们定义模型学习到的均值 $\mu_\theta$ 为一个包含神经网络 $\epsilon_\theta$ 的函数。 $\epsilon_\theta(\mathbf{x}_t, t)$ 就是我们的神经网络，它以当前的含噪数据 $\mathbf{x}_t$ 和时间步 $t$ 作为输入，其输出是一个对原始噪声 $\epsilon$ 的预测。 这个公式表明，一旦我们的网络预测出了噪声 $\epsilon_\theta$，我们就可以用这个公式来构造出反向过程所需的均值 $\mu_\theta$。

>   神经网络的任务是从含噪的 $\mathbf{x}_t$ 中预测出**原始**噪声 $\epsilon$，$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$ 这个等式能很好的帮助我们理解这一点，其中的 $\bar{\alpha}_t$ 就是一个与时间 $t$ 有关的量。

前面我们提到，模型的形式为：

$$
p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \sigma_t^2\mathbf{I})
$$

当要从 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 中采样 $\mathbf{x}_{t-1}$ 时，首先将 $\mathbf{x}_t$ 和 $t$ 输入神经网络，得到噪声预测 $\epsilon_\theta(\mathbf{x}_t, t)$，而后进一步得到去噪后的均值 $\mu_\theta$，最后加上一个新的、方差为 $\sigma_t^2$ 的高斯噪声，就得到 $\mathbf{x}_{t-1}$。即 $\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_\theta(\mathbf{x}_t, t)\right) + \sigma_t\mathbf{z}$，其中 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。从 $t = T$ 开始重复这个过程 $T$ 次，就可以从纯噪声生成一个清晰的样本。完整的采样流程如下图所示：

<div align="center"><img src="https://wkqpicture.oss-cn-beijing.aliyuncs.com/img/20250606221800191.png" width="50%" alt="image-20250606221749848"></div>

其中，只有在 $t>1$ 时 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$，$t=0$ 时 $\mathbf{z}=\mathbf{0}$，其目的是为了**让中间的去噪步骤是随机的，而最后一步（生成最终图像）是确定性的**。即：

-   经过了 $T - 1$ 步的随机去噪，我们已经得到了一个非常接近真实图像的 $\mathbf{x}_1$。
-   在这最后一步，模型给出了关于最终清晰图像 $\mathbf{x}_0$ 的一个概率分布。与其从这个分布中再进行一次随机采样（这可能会给最终图像增加一层微小的、不必要的噪声或“颗粒感”），作者选择直接取这个**分布的均值**。
-   在概率分布中，均值通常代表了“最可能”或“期望”的结果。因此，将均值作为最终输出，可以得到一个更稳定、更清晰的图像，可以看作是模型对最终结果的“最佳估计”。

>   这里额外总结一下：
>
>   1.   反向过程的时间流：采样（生成）过程是反向的，时间步 $t$ 从 $T$ 开始，一步步递减，直到 $t = 1$。
>   2.   前向过程 $\beta_t$ 的设计：$\beta_t$ （前向加噪的方差）是一个随下标 $t$ 增大而增大的序列（例如，DDPM使用的线性调度）。所以，$\beta_1 < \beta_2 < \cdots < \beta_{T-1} < \beta_T$。
>   3.   反向过程噪声方差 $\sigma_t^2$ 的选择：DDPM论文中的一个关键设计选择就是将反向过程每一步加入的噪声方差设定为：$\sigma_t^2 = \beta_t$。
>   4.   推论：反向过程中加入的噪声逐渐减小，直到最后一步添加噪声为 $0$。
>
>   这个设计非常直观，可以做一个比喻： 想象你在修复一幅被严重污染的画作（从噪声 $\mathbf{x}_T$ 开始）。
>
>   -   初期 ($t$ 很大): 画作非常模糊，你可能会用大笔触（对应大的噪声方差 $\sigma_T^2$）来勾勒出大的轮廓和结构，允许一些随机性和不确定性。
>   -   中期 ($t$ 减小): 轮廓逐渐清晰，你会换成小一些的笔刷（对应减小的 $\sigma_t^2$），进行更精细的修复，同时仍然保留一些随机性来丰富细节。
>   -   最后一步 ($t = 1$): 画作已经基本修复完成（对应 $\mathbf{x}_1$），只剩下最后的“点睛之笔”。这时，你不会再随机地往上泼洒颜料，而是会用最精细的笔触，做出最确定的描绘（对应 $\mathbf{z} = \mathbf{0}$，只取均值），以确保最终成品是最清晰、最稳定的。

以上的采样过程类似于朗之万动力学，这是一种基于物理学原理的采样方法。它通过模拟粒子在势能场中的运动来进行采样，粒子在运动过程中，一方面沿着概率密度的对数梯度方向移动，以趋向高概率区域，另一方面会添加一些随机噪声，避免粒子陷入局部最优。 在DDPM相关内容中，其采样过程与朗之万动力学相似。在DDPM的反向采样（生成）过程里，从含噪数据生成清晰样本时，将含噪数据 $\mathbf{x}_t$ 和时间步 $t$ 输入神经网络得到噪声预测 ，再基于此计算出去噪后的均值 ，并添加新噪声得到新数据 ，重复此过程。这里预测的噪声与朗之万动力学中的梯度（分数score）有直接关系，使得整个过程类似于朗之万动力学从概率分布中采样的方式。 

进一步的，将公式 $\eqref{eq:mu}$ 代入公式 $\eqref{eq:l-c}$，容易得到：

$$
\begin{equation}
\boxed{ \mathbb{E}_{\mathbf{x}_0, \epsilon}\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1 - \bar{\alpha}_t)}\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, t)\|^2\right] } \label{eq:e1}
\end{equation}
$$

这个推导过程是DDPM论文的画龙点睛之笔。它揭示了：

-   损失函数的简化：一个原本非常复杂的、包含KL散度的损失项 $L_{t-1}$，被等价地转换成了一个非常简单的均方误差 (MSE)： $\|\epsilon - \epsilon_\theta(\mathbf{x}_t, t)\|^2$。
-   模型任务的简化：模型的任务从预测一个复杂的条件均值 $\tilde{\mu}_t$，转变为预测用于生成 $\mathbf{x}_t$ 的原始噪声 $\epsilon$。这在概念上更清晰，在实践中也更稳定有效。
-   加权系数：前面的分数 $\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1 - \bar{\alpha}_t)}$ 是一个只依赖于时间步 $t$ 的权重。它会根据不同的噪声水平（不同的 $t$）调整损失的重要性。
-   训练过程的简化：整个训练过程变得非常直观：
    1.   取一个干净样本 $\mathbf{x}_0$。
    2.   **随机**选择一个时间步 $t$。
    3.   生成一个随机噪声 $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。
    4.   用公式 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$ 构造出含噪样本 $\mathbf{x}_t$。
    5.   将 $\mathbf{x}_t$ 和 $t$ 输入神经网络，得到噪声预测 $\epsilon_\theta(\mathbf{x}_t, t)$。
    6.   计算真实噪声 $\epsilon$ 和预测噪声 $\epsilon_\theta$ 之间的均方误差，并根据这个误差更新网络参数。

其训练流程如下所示，后续还会进一步介绍：

<div align="center"><img src="https://wkqpicture.oss-cn-beijing.aliyuncs.com/img/20250607130703736.png" width="50%" alt="image-20250607130655511"></div>

## （三）数据缩放、反向过程解码器与 $L_0$

图像像素数据由 $\{0, 1, \ldots, 255\}$ 中的整数组成，我们将原始图像数据线性缩放到 $[-1, 1]$ 的区间内。这样做的动机是**尺度一致性**。反向生成过程是从一个标准正态分布 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 中采样的噪声 $\mathbf{x}_T$ 开始的，这个分布的数值集中在 $0$ 附近，可以使得整个去噪过程（从 $\mathbf{x}_T$ 到 $\mathbf{x}_0$）都在一个稳定的数值范围内进行，这有利于神经网络的**训练**。（被缩放的只有原始数据 $\mathbf{x}_0$ 和最后一步由 $\mathbf{x}_1$ 生成 $\mathbf{x}_0$ 时，从 $\mathbf{x}_T$ 到 $\mathbf{x}_1$ 的中间结果不进行缩放）

为了获得离散的对数似然，我们将反向过程的最后一步设定为一个独立的**离散解码器**。模型经过 $T - 1$ 步去噪后，在最后一步预测的是一个连续的高斯分布 $\mathcal{N}(\mathbf{x}_0; \mu_\theta(\mathbf{x}_1, 1), \sigma_1^2\mathbf{I})$，其均值 $\mu_\theta(\mathbf{x}_1, 1)$ 是一个在 $[-1, 1]$ 区间内的连续值（从连续区间内选定的值）。然而，真实的图像数据是离散的，对于一个连续的概率分布，任何单个离散点的概率都为零。因此，我们无法直接计算 $p_\theta(\mathbf{x}_0|\mathbf{x}_1)$。为此，需要设计一个解码器，将模型输出的连续分布转换成在离散数据空间上的概率：

$$
\begin{equation}
\boxed{ p_\theta(\mathbf{x}_0|\mathbf{x}_1) = \prod_{i=1}^{D}\int_{\delta_-(x_0^i)}^{\delta_+(x_0^i)}\mathcal{N}(x; \mu_\theta^i(\mathbf{x}_1, 1), \sigma_1^2)dx } \label{eq:e2}
\end{equation}
$$

$$
\begin{equation}
\delta_+(x) = \begin{cases}
\infty & \text{if } x = 1 \\
x + \frac{1}{255} & \text{if } x < 1
\end{cases}

~~~~~~
\delta_-(x) = \begin{cases}
-\infty & \text{if } x = -1 \\
x - \frac{1}{255} & \text{if } x > -1
\end{cases}
\end{equation}
$$

其中，$D$ 是数据维度，$i$ 是像素点索引。上式的核心思想是：计算模型预测的连续高斯分布落在这个离散值所代表的区间内的概率。

- $\prod_{i=1}^{D}$: **假设像素之间互相独立**，整张图像的概率是就是每个像素 $i$ 的概率的连乘积。
- $\int_{\delta_-(x_0^i)}^{\delta_+(x_0^i)} \ldots dx$: 对于第 $i$ 个像素，其真实的离散值是 $x_0^i$（已经缩放到[-1, 1]）。我们通过对模型预测的一维高斯分布 $\mathcal{N}(x; \mu_\theta^i(\mathbf{x}_1, 1), \sigma_1^2)$ 在一个小的区间 $[\delta_-(x_0^i), \delta_+(x_0^i)]$ 上进行积分，来得到这个离散值的概率。
- $\delta$ 函数定义的区间：
    - 对于中间值 (例如，1到254的像素值)：每个离散值代表一个宽度为 $2/255$ 的区间。例如，对于缩放后的值 $x_0^i$，其区间是 $[x_0^i - \frac{1}{255}, x_0^i + \frac{1}{255}]$。
    - 对于边界值 (0或255)：
        - 如果是像素值255（缩放后为1），我们计算高斯分布落在 $[1 - \frac{1}{255}, \infty)$ 这个区间的概率。
        - 如果是像素值0（缩放后为-1），我们计算高斯分布落在 $(-\infty, -1 + \frac{1}{255}]$ 这个区间的概率。
- 这个过程确保了整个连续高斯分布的概率被完整地分配给了256个离散的像素值。

**像素独立的假设是一种简化，未来可以使用更强大的解码器来提升性能**。

在**采样**结束时，我们**无噪声**地显示 $\mu_\theta(\mathbf{x}_1, 1)$。因此，这里需要注意的细节是，最后一步要区分训练与生成：

-   为了计算损失 $L_0$（**训练时**）：我们使用上面定义的、复杂的积分解码器来计算离散数据的精确对数似然。
-   为了生成图像（**采样时**）：我们并不执行这个积分过程。我们采用算法2的确定性最后一步，即 $\mathbf{x}_0 = \mu_\theta(\mathbf{x}_1, 1)$。这个结果是一个连续值（在[-1, 1]内），我们直接将其缩放回[0, 255]并取整，作为最终的生成图像来显示。这样做是为了获得视觉上最清晰、最平滑的图像。

## （四）简化训练目标

通过上面定义好的反向过程和解码器，由公式 $\eqref{eq:e1}$ 和 $\eqref{eq:e2}$ 推导出的变分界限，可以用于训练。然而，我们发现，训练下面这个变分界限的变体，对样本质量更有益（并且实现起来更简单）：

$$
\begin{equation}
\boxed{ L_{\text{simple}}(\theta) := \mathbb{E}_{t, \mathbf{x}_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, t)\|^2 \right] }
\end{equation}
$$

这是**DDPM最终采用的核心损失函数**。这个损失函数的期望是针对三个随机变量来计算的：

1.   $\mathbf{x}_0$：从真实数据集中随机抽取的一个样本。
2.   $t$：从 $\{1, \ldots, T\}$ 中**均匀随机**抽取的一个时间步。
3.   $\epsilon$：从标准正态分布 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 中随机抽取的一个噪声。

损失的核心 $\|\epsilon - \epsilon_\theta(\mathbf{x}_t, t)\|^2$ 是均方误差 (Mean Squared Error, MSE)。它计算的是真实加入的噪声 $\epsilon$ 和我们的神经网络 $\epsilon_\theta$ 从含噪图像 $\mathbf{x}_t$ 中预测出的噪声之间的差异。这个公式与我们之前推导出的公式 $\eqref{eq:e1}$ 相比，最显著的变化是舍弃了那个复杂的加权系数 $\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1 - \bar{\alpha}_t)}$。它变成了一个**无加权**的均方误差。

其中，$t = 1$ 的情况对应 $L_0$，此时离散解码器定义中的积分由**高斯概率密度函数乘以区间宽度近似**，忽略 $\sigma_1^2$ 和边缘效应。$t > 1$ 的情况对应公式 $\eqref{eq:e1}$ 的非加权版本，类似于 NCSN 去噪得分匹配模型中使用的损失加权方式。$L_T$ 未出现，正如前文所说，因为前向过程固定，所以 $L_T$ 是一个与模型参数无关的常数，在训练中被忽略了。前文的算法 1 展示了采用这种简化目标的完整训练流程。

我们关注公式 $\eqref{eq:e1}$ 中的权重：

$$
W_t = \frac{\beta_t^2}{2\sigma_t^2\alpha_t(1 - \bar{\alpha}_t)}
$$

论文中为了简化，做出了关键选择 $\sigma_t^2 = \beta_t$。同时我们知道 $\alpha_t = 1 - \beta_t$。将这些代入权重表达式：

$$
W_t = \frac{\beta_t^2}{2(\beta_t)\alpha_t(1 - \bar{\alpha}_t)} = \frac{1 - \alpha_t}{2\alpha_t(1 - \bar{\alpha}_t)}
$$


为了分析 $W_t$ 的变化，我们需要看其分子和分母中各项的变化趋势，我们使用论文中的线性调度，其中 $\beta_t$ 从 $\beta_1 = 10^{-4}$ 线性增加到 $\beta_T = 0.02 ~~(T = 1000)$。由于分子和分母都有增大和减小的项，我们通过代码来画出 $W_t$ 的变化曲线：

<div align="center"><img src="https://wkqpicture.oss-cn-beijing.aliyuncs.com/img/20250607165327149.png" width="100%" alt="image-20250607165325921"></div>

因此，这一简化目标舍弃了公式 $\eqref{eq:e1}$ 中的权重，它实际上是一个重新加权的变分界限，与标准的变分界限相比，它强调了重构的不同方面。通过采用均匀的权重，相比理论权重，会降低 $t$ 较小的时候的损失权重，这种降低是有益的，因为这样神经网络就可以专注于在较大的 $t$ 时更困难的去噪任务。

# 三、实验部分

实验部分就不详细记录了，大致总结一下：

**模型设置**

用于近似噪声 $\epsilon_\theta(\mathbf{x}_t, t)$ 的神经网络的具体架构：

1.   核心架构：
     -   采用了 U-Net 架构。U-Net 是一种带有跳跃连接 (skip connections) 的对称编码器-解码器网络。这种结构非常适合图像到图像的任务，因为它既能捕捉深层的语义信息（通过编码器下采样），又能保留高分辨率的细节信息（通过跳跃连接将编码器特征传递给解码器）。
2.   时间步的处理：
     -   参数共享：作者只使用一个 U-Net 网络来处理所有 $T = 1000$ 个时间步，而不是为每个时间步训练一个单独的模型。这大大提高了参数效率。
     -   时间嵌入：为了让这个共享的网络知道当前正在为哪个时间步 $t$ 进行去噪，作者使用了 Transformer 模型中的正弦位置嵌入技术。它将一个整数时间步 $t$ 转换为一个高维的向量。这个时间嵌入向量随后被整合到U-Net的残差块中，从而让网络的行为能够根据当前的时间步进行调整。
3.   关键模块：
     -   自注意力机制：在 U-Net 的特征图分辨率被降到较低的 $16 \times 16$ 时，作者加入了一个自注意力模块。这使得模型能够捕捉图像中不同区域之间的长距离依赖关系，从而更好地理解和生成具有全局一致性的结构。
     -   归一化：整个网络使用组归一化 (Group Normalization) 来稳定训练过程。

**主要结论与贡献**

主要体现在以下几点：

1.  图像生成质量达到新高度
    -   在衡量生成图像质量的两个关键指标——**Inception Score (IS)** 和 **Fréchet Inception Distance (FID)** 上，DDPM在多个数据集（如CIFAR-10, CelebA-HQ等）上**全面超越了当时所有的生成模型**，包括最强的GAN。
    -   特别是**FID分数**，它被认为是衡量图像真实性和多样性的黄金标准。DDPM在该指标上取得了当时最低（最好）的记录，证明其生成的图像在视觉上与真实图像非常难以区分。
2.  似然估计具有竞争力
    -   在衡量模型对数据分布拟合程度的指标——**负对数似然 (NLL)** 上，DDPM也取得了具有竞争力的结果。
    -   虽然它的NLL分数没有超过专门为此优化的自回归模型（如PixelCNN++），但它击败了大多数其他类型的生成模型（如VAE和GAN），证明了其作为一种优秀的似然模型的潜力。

**消融实验**

这部分实验验证了作者在论文中提出的简化方案确实是有效的。

1.  简化损失函数 $L_{\text{simple}}$ 的优越性
    -   对比两种训练目标
        1.  预测噪声 $\epsilon$ (使用简化的 $L_{\text{simple}}$ 损失函数)
        2.  预测均值 $\tilde\mu_t$ (使用理论上更复杂的 $L_{t−1}$ 损失函数)
    -   结论：实验结果明确表明，预测噪声 $\epsilon$ 的方案显著优于直接预测均值 $\tilde{\mu}_t$。这证明了他们提出的简化损失函数不仅实现简单，而且在实践中效果更好。这是DDPM论文最重要的实践贡献之一。
2.  反向过程方差 $\sigma^2$ 的选择
    -   对比：作者比较了两种为反向过程固定方差的策略：$\sigma^2=\beta_t$ 和 $\sigma^2=\tilde\beta_t$。
    -   结论：实验发现这两种选择产生的**结果非常相似**。因此，选择更简单的 $\sigma^2=\beta_t$ 是完全合理的。

**模型行为的可视化与分析**

1.  渐进式生成过程
    -   论文展示了从纯噪声 $\mathbf{x}_T$ 开始，经过1000步反向去噪，最终生成清晰图像 $\mathbf{x}_0$ 的完整过程。
    -   我们可以清晰地看到，图像是**逐步、分层地**被生成的：最开始出现的是模糊的轮廓和全局结构，然后逐渐增加中层细节，最后再补完精细的纹理。这个过程非常直观地展示了扩散模型的工作原理。
2.  平滑的插值能力
    -   实验还表明，DDPM可以在潜空间中进行非常平滑的插值。通过对两个不同图像的初始噪声 $\mathbf{x}_T$ 进行线性插值，然后进行反向生成，可以得到介于这两张图像之间的、平滑过渡的全新图像。


# 参考链接

1.   [扩散模型(Diffusion Model)奠基之作：DDPM 论文解读 - 知乎](https://zhuanlan.zhihu.com/p/682840224)
2.   [Denoising Diffusion Probabilistic Models 全过程概述 + 论文总结-CSDN博客](https://blog.csdn.net/qq_45981086/article/details/139097700)







