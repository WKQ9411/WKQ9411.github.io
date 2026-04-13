---
layout: post
title: 【强化学习】10-Actor-Critic
date: 2026-03-22 16:00:00 +08:00
summary: 此系列是西湖大学赵世钰老师《强化学习的数学原理》笔记。
categories: RL
excerpt_separator: <!--more-->
---

{% raw %}
# 目录
{:.no_toc}

* TOC
{:toc}

---

Actor-Critic 方法本质上仍然是策略梯度方法，它强调的是融合了策略梯度和基于值的方法。

什么是 "actor" 和 "critic"？

- 这里的 "actor" 指的是策略更新。之所以称之为 actor，是因为策略将被用来执行动作。
- 这里的 "critic" 指的是策略评估或价值估计。之所以称之为 critic，是因为它通过评估策略来"评判"策略的好坏。

# 一、最简单的 Actor-Critic (QAC)

回顾上一章介绍的策略梯度的核心思想。

1. 一个标量度量 $J ( \theta )$，可以是 $\bar { v } _ { \pi }$ 或 $\bar { r } _ { \pi }$。

2. 最大化 $J ( \theta )$ 的梯度上升算法为：
   $$
   \begin{equation}
   \theta_ {t + 1} = \theta_ {t} + \alpha \nabla_ {\theta} J (\theta_ {t}) = \theta_ {t} + \alpha \mathbb {E} _ {S \sim \eta , A \sim \pi} \Big [ \nabla_ {\theta} \ln \pi (A | S, \theta_ {t}) q _ {\pi} (S, A) \Big ]
   \end{equation}
   $$

3. 对应的随机梯度上升算法为：
   $$
   \begin{equation}
   \theta_ {t + 1} = \theta_ {t} + \alpha \nabla_ {\theta} \ln \pi \left(a _ {t} \mid s _ {t}, \theta_ {t}\right) {\color{blue}{q _ {t} \left(s _ {t}, a _ {t}\right)}}
   \end{equation}
   $$

从这个算法中我们可以看出 "actor" 和 "critic" 的角色：

- 这个算法本身对应的是 actor，因为它在更新策略的参数。
- 而估计 $q _ { t } ( s , a )$ 的算法对应的是 critic。

如何得到 $q _ { t } ( s _ { t } , a _ { t } )$？

到目前为止，我们学习了两种估计动作值的方法：
• **蒙特卡洛学习**：如果使用 MC，相应的算法称为 REINFORCE 或蒙特卡洛策略梯度。（上一章已介绍）
• **时序差分学习**：如果使用 TD，这类算法通常称为 actor-critic。（本章将介绍）

最简单的 Actor-Critic (QAC) 伪代码如下：

$$
\begin{equation*}
\begin{aligned}
& \text{Aim: Search for an optimal policy maximizing } J(\theta). \\
& \text{At time step } t \text{ in episode, do} \\
& \quad \quad \text{Generate } a_t \text{ following } \pi(a|s_t,\theta_t), \text{ observe } r_{t+1}, s_{t+1}, \text{ and then generate } a_{t+1} \\
& \quad \quad \text{following } \pi(a|s_{t+1},\theta_t). \text{ (Sarsa algorithm) } \\
& \quad \quad \text{Critic (value update): } \\
& \quad \quad \quad \quad w _ {t + 1} = w _ {t} + \alpha_ {w} [ r _ {t + 1} + \gamma q (s _ {t + 1}, a _ {t + 1}, w _ {t}) - q \left(s _ {t}, a _ {t}, w _ {t}\right) ] \nabla_ {w} q \left(s _ {t}, a _ {t}, w _ {t}\right) \\
& \quad \quad \text{Actor (policy update): } \\
& \quad \quad \quad \quad \theta_ {t + 1} = \theta_ {t} + \alpha_ {\theta} \nabla_ {\theta} \ln \pi \left(a _ {t} \mid s _ {t}, \theta_ {t}\right) q \left(s _ {t}, a _ {t}, w _ {t + 1}\right)
\end{aligned}
\end{equation*}
$$

几点说明：

- Critic 对应的是 "SARSA+值函数近似"。
- Actor 对应的是策略更新算法。
- 该算法是 **on-policy** 的。
- 由于策略是随机的，且 $\pi (a | s, \theta) > 0$，因此无需使用像 ε-greedy 这样的技巧。
- 这个特定的 actor-critic 算法有时被称为 **Q Actor-Critic (QAC)**。
- 尽管简单，但这个算法揭示了 actor-critic 方法的核心思想，可以扩展成后续介绍的许多其他算法。

# 二、Advantage Actor-Critic (A2C)

A2C 的核心思想是**引入一个基线（baseline）来减小方差**。

## 1. 基线的不变性

有这样一个性质，策略梯度在减去一个基线函数后保持不变：
$$
\begin{equation}
\nabla_ {\theta} J (\theta) = \mathbb {E} _ {S \sim \eta , A \sim \pi} \Big [ \nabla_ {\theta} \ln \pi (A | S, \theta_ {t}) {\color{blue}{q _ {\pi} (S, A)}} \Big ] = \mathbb {E} _ {S \sim \eta , A \sim \pi} \left[ \nabla_ {\theta} \ln \pi (A | S, \theta_ {t}) {\color{blue}{\left(q _ {\pi} (S, A) - b (S)\right)}} \right]
\end{equation}
$$
这里，额外的基线 $b ( S )$ 是一个只关于状态 $S$ 的标量函数。

**首先，为什么不变？**

这是因为添加的基线项的期望为零：
$$
\begin{equation}
\mathbb {E} _ {S \sim \eta , A \sim \pi} \left[ \nabla_ {\theta} \ln \pi (A | S, \theta_ {t}) b (S) \right] = 0
\end{equation}
$$
详细推导如下：
$$
\begin{equation}
\begin{aligned}
\mathbb {E} _ {S \sim \eta , A \sim \pi} \left[ \nabla_ {\theta} \ln \pi (A | S, \theta_ {t}) b (S) \right] &= \sum_ {s \in \mathcal {S}} \eta (s) \sum_ {a \in \mathcal {A}} \pi (a | s, \theta_ {t}) \nabla_ {\theta} \ln \pi (a | s, \theta_ {t}) b (s) \\ 
&= \sum_ {s \in \mathcal {S}} \eta (s) \sum_ {a \in \mathcal {A}} \nabla_ {\theta} \pi (a | s, \theta_ {t}) b (s) \\ 
&= \sum_ {s \in \mathcal {S}} \eta (s) b (s) \sum_ {a \in \mathcal {A}} \nabla_ {\theta} \pi (a | s, \theta_ {t}) \\ 
&= \sum_ {s \in \mathcal {S}} \eta (s) b (s) \nabla_ {\theta} \sum_ {a \in \mathcal {A}} \pi (a | s, \theta_ {t}) \\ 
&= \sum_ {s \in \mathcal {S}} \eta (s) b (s) \nabla_ {\theta} 1 = 0
\end{aligned}
\end{equation}
$$

**其次，为什么基线有用？**

梯度可以看作是 $\nabla _ { \theta } J ( \theta ) = \mathbb { E } [ X ]$，其中：
$$
\begin{equation}
X (S, A) \dot {=} \nabla_ {\theta} \ln \pi (A | S, \theta_ {t}) [ q _ {\pi} (S, A) - b (S) ]
\end{equation}
$$
我们已经知道 $\mathbb {E}[X]$ 对于 $b ( S )$ 是不变的，然而，$\operatorname{var}(X)$ 对于 $b ( S )$ 是**变化**的。为什么？

令 $\bar{x} \doteq \mathbb{E}[X]$，它对于任意 $b(s)$ 都是不变的。如果 $X$ 是**向量**，其方差是一个**矩阵**。通常选择 $\mathrm{var}(X)$ 的迹作为优化的标量目标函数：

$$
\begin{equation}
\begin{aligned}
\mathrm{tr}[\mathrm{var}(X)] &= \mathrm{tr}\mathbb{E}[(X - \bar{x})(X - \bar{x})^T] \\
&= \mathrm{tr}\mathbb{E}[XX^T - \bar{x}X^T - X\bar{x}^T + \bar{x}\bar{x}^T] \\
&= \mathrm{tr}\mathbb{E}[XX^T] - \mathrm{tr}\mathbb{E}[\bar{x}X^T] - \mathrm{tr}\mathbb{E}[X\bar{x}^T] + \mathrm{tr}\mathbb{E}[\bar{x}\bar{x}^T] \\
&= \mathbb{E}[X^T X - X^T \bar{x} - \bar{x}^T X + \bar{x}^T \bar{x}] \\
&= \mathbb{E}[X^T X] - \mathbb{E}[X]^T \bar{x} - \bar{x}^T\mathbb{E}[X] + \bar{x}^T \bar{x} \\
&= \mathbb{E}[X^T X] - \bar{x}^T \bar{x} - \bar{x}^T\bar{x} + \bar{x}^T \bar{x} \\
&= \mathbb{E}[X^T X] - \bar{x}^T \bar{x}
\end{aligned} \label{eq:var}
\end{equation}
$$

在推导上述公式时，我们使用了如下的恒等式：
$$
\begin{equation}
\mathrm{tr}(ab^T) = b^T a
\end{equation}
$$
其中 $a,b$ 是列向量。

由于 $\bar{x}$ 是不变的，公式 $\eqref{eq:var}$ 表明我们只需最小化 $\mathbb{E}[X^T X]$，有：
$$
\begin{equation}
\begin{aligned}
\mathbb{E}[X^T X] &= \mathbb{E}\left[(\nabla_\theta \ln \pi)^T (\nabla_\theta \ln \pi)(q_\pi(S, A) - b(S))^2\right] \\
&= \mathbb{E}\left[\|\nabla_\theta \ln \pi\|^2 (q_\pi(S, A) - b(S))^2\right]
\end{aligned}
\end{equation}
$$

其中 $\pi(A|S, \theta)$ 简写为 $\pi$。由于 $S \sim \eta$ 且 $A \sim \pi$，上式可重写为：

$$
\begin{equation}
\mathbb{E}[X^T X] = \sum_{s \in \mathcal{S}} \eta(s) \mathbb{E}_{A \sim \pi} \left[\|\nabla_\theta \ln \pi\|^2 (q_\pi(s, A) - b(s))^2\right]
\end{equation}
$$

为确保 $\nabla_b \mathbb{E}[X^T X] = 0$，对任意 $s \in \mathcal{S}$，$b(s)$ 应满足：

$$
\begin{equation}
\mathbb{E}_{A \sim \pi} \left[\|\nabla_\theta \ln \pi\|^2 (b(s) - q_\pi(s, A))\right] = 0, \quad s \in \mathcal{S}
\end{equation}
$$

该方程可轻易求解，得到最优基线：

$$
\begin{equation}
b^*(s) = \frac{\mathbb{E}_{A \sim \pi}[\|\nabla_\theta \ln \pi\|^2 q_\pi(s, A)]}{\mathbb{E}_{A \sim \pi}[\|\nabla_\theta \ln \pi\|^2]}, \quad s \in \mathcal{S}
\end{equation}
$$

这样一来，当我们采用最优基线并通过随机样本近似 $\mathbb { E } [ X ]$ 时，估计的方差也会很小。在 REINFORCE 和 QAC 算法中，可以视为没有使用基线，我们可以认为 $b = 0$，但这并不能保证是一个好的基线。

尽管我们上面推导出的这个基线是最优的，但它形式复杂。我们可以去掉权重 $\| \nabla _ { \theta } \ln \pi ( A | s , \theta _ { t } ) \| ^ { 2 }$，选择一个**次优基线**：
$$
\begin{equation}
b (s) = \mathbb {E} _ {A \sim \pi} [ q _ {\pi} (s, A) ] = v _ {\pi} (s)
\end{equation}
$$
这正是状态 $s$ 的**状态值**。

## 2. A2C 算法

当取基线 $b ( s ) = v _ { \pi } ( s )$ 时，梯度上升算法变为：
$$
\begin{equation}
\begin{aligned}
\theta_ {t + 1} &= \theta_ {t} + \alpha \mathbb {E} \left[ \nabla_ {\theta} \ln \pi (A | S, \theta_ {t}) [ q _ {\pi} (S, A) - v _ {\pi} (S) ] \right] \\
&= \theta_ {t} + \alpha \mathbb {E} \left[ \nabla_ {\theta} \ln \pi (A | S, \theta_ {t}) \delta_ {\pi} (S, A) \right]
\end{aligned}
\end{equation}
$$
其中：
$$
\begin{equation}
\delta_ {\pi} (S, A) \dot {=} q _ {\pi} (S, A) - v _ {\pi} (S)
\end{equation}
$$
被称为**优势函数**（$\delta$ 通常用 $A$ 来表示，但这里因为已经有 $A$ 了，因此用 $\delta$）。由于 $v _ {\pi} = \sum_a \pi(a|s) q_\pi$，它是动作值的平均，因此 $\delta$ 表达了一个动作对平均动作的优势。

对应的随机版本算法为：
$$
\begin{equation}
\begin{aligned}
\theta_ {t + 1} &= \theta_ {t} + \alpha \nabla_ {\theta} \ln \pi \left(a _ {t} \mid s _ {t}, \theta_ {t}\right) \left[ q _ {t} \left(s _ {t}, a _ {t}\right) - v _ {t} \left(s _ {t}\right) \right] \\
&= \theta_ {t} + \alpha \nabla_ {\theta} \ln \pi (a _ {t} | s _ {t}, \theta_ {t}) \delta_ {t} (s _ {t}, a _ {t})
\end{aligned}
\end{equation}
$$

进一步，该算法可以重新表示为：
$$
\begin{equation}
\begin{aligned}
\theta_{t + 1} &= \theta_{t} + \alpha \nabla_{\theta} \ln \pi \left(a_{t} | s_{t}, \theta_{t}\right) \delta_{t} \left(s_{t}, a_{t}\right) \\
&= \theta_{t} + \alpha \frac{\nabla_{\theta} \pi \left(a_{t} | s_{t} , \theta_{t}\right)}{\pi \left(a_{t} | s_{t} , \theta_{t}\right)} \delta_{t} \left(s_{t}, a_{t}\right) \\
&= \theta_{t} + \alpha \underbrace{\left(\frac{\delta_{t} (s_{t} , a_{t})}{\pi (a_{t} | s_{t} , \theta_{t})}\right)}_{\text{step size}} \nabla_{\theta} \pi (a_{t} | s_{t}, \theta_{t})
\end{aligned}
\end{equation}
$$
这与上一章所介绍的平衡探索与利用的形式相同，只不过这里步长与相对值 $\delta _ { t }$ 成正比，而不是绝对值 $q _ { t }$，这更加合理。

更进一步，**优势函数可以用 TD 误差来近似**：
$$
\begin{equation}
\delta_ {t} = q _ {t} \left(s _ {t}, a _ {t}\right) - v _ {t} \left(s _ {t}\right)\rightarrow r _ {t + 1} + \gamma v _ {t} \left(s _ {t + 1}\right) - v _ {t} \left(s _ {t}\right)
\end{equation}
$$
这种近似是合理的，因为：
$$
\begin{equation}
\mathbb {E} [ q _ {\pi} (S, A) - v _ {\pi} (S) | S = s _ {t}, A = a _ {t} ] = \mathbb {E} \Big [ R + \gamma v _ {\pi} (S ^ {\prime}) - v _ {\pi} (S) | S = s _ {t}, A = a _ {t} \Big ]
\end{equation}
$$
这样做的好处是：只需要一个网络来近似 $v _ { \pi } ( s )$，而不是分别用两个网络来近似 $q _ { \pi } ( s , a )$ 和 $v _ { \pi } ( s )$。

A2C 的伪代码如下：

$$
\begin{equation*}
\begin{aligned}
& \text{Aim: Search for an optimal policy maximizing } J(\theta). \\
& \text{At time step } t \text{ in episode, do} \\
& \quad \quad \text{Generate } a_t \text{ following } \pi(a|s_t,\theta_t), \text{ and then observe } r_{t+1}, s_{t+1}. \\
& \quad \quad \text{TD error (advantage function): } \\
& \quad \quad \quad \quad \delta_ {t} = r _ {t + 1} + \gamma v \left(s _ {t + 1}, w _ {t}\right) - v \left(s _ {t}, w _ {t}\right) \\
& \quad \quad \text{Critic (value update): } \\
& \quad \quad \quad \quad w _ {t + 1} = w _ {t} + \alpha_ {w} \delta_ {t} \nabla_ {w} v \left(s _ {t}, w _ {t}\right) \\
& \quad \quad \text{Actor (policy update): } \\
& \quad \quad \quad \quad \theta_ {t + 1} = \theta_ {t} + \alpha_ {\theta} \delta_ {t} \nabla_ {\theta} \ln \pi \left(a _ {t} | s _ {t}, \theta_ {t}\right)
\end{aligned}
\end{equation*}
$$

该算法是 on-policy 的，由于策略 $\pi ( \theta _ { t } )$ 是随机的，无需使用像 ε-greedy 这样的技巧。

# 三、重要性采样和 off-policy Actor-Critic

策略梯度是 **on-policy** 的：因为梯度表达式为 $\nabla _ { \theta } J ( \theta ) = \mathbb { E } _ { S \sim \eta , {\color{blue}{A \sim \pi}} } [ * ]$，其中 $A$ 服从 $\pi$，因此采样时，$\pi$ 是一个 behavior policy，同时，$\pi$ 又是被优化的。

我们能否将其转换为 **off-policy**？可以，通过使用**重要性采样**技术。重要性采样技术不仅限于 AC，也适用于任何旨在估计期望的算法。

## 1. 例子

考虑一个随机变量 $X \in { \mathcal { X } } = \{ + 1 , - 1 \}$。如果 $X$ 的概率分布为 $p _ { 0 }$：
$$
\begin{equation}
p _ {0} (X = + 1) = 0. 5, \quad p _ {0} (X = - 1) = 0. 5
\end{equation}
$$
那么 $X$ 的期望是：
$$
\begin{equation}
\mathbb {E} _ {X \sim p _ {0}} [ X ] = (+ 1) \cdot 0. 5 + (- 1) \cdot 0. 5 = 0
\end{equation}
$$
问题：如何使用一些样本 $\{ x _ { i } \}$ 来估计 $\mathbb { E } [ X ]$？

**情形 1**：
样本 $\{ x _ { i } \}$ 根据 $p _ { 0 }$ 生成。那么，样本平均值会收敛到期望值：
$$
  \bar {x} = \frac {1}{n} \sum_ {i = 1} ^ {n} x _ {i} \rightarrow \mathbb {E} [ X ], \quad \text {as } n \rightarrow \infty
$$

**情形 2**：
样本 $\{ x _ { i } \}$ 根据另一个分布 $p_1$ 生成：
$$
\begin{equation}
p _ {1} (X = + 1) = 0. 8, \quad p _ {1} (X = - 1) = 0. 2
\end{equation}
$$
其期望为：
$$
\begin{equation}
\mathbb {E} _ {X \sim p _ {1}} [ X ] = (+ 1) \cdot 0. 8 + (- 1) \cdot 0. 2 = 0. 6
\end{equation}
$$
如果我们直接使用这些样本的平均值，那么：
$$
\begin{equation}
\bar {x} = \frac {1}{n} \sum_ {i = 1} ^ {n} x _ {i} \rightarrow \mathbb {E} _ {X \sim p _ {1}} [ X ] = 0. 6 \neq \mathbb {E} _ {X \sim p _ {0}} [ X ]
\end{equation}
$$

问题：我们能否使用来自 $p _ { 1 }$ 的样本 $\{ x _ { i } \}$ 来估计 $\mathbb{E}_{X\sim p_0}[X]$？为什么这样做？因为我们可能想根据行为策略 $\beta$ 的样本来估计关于目标策略 $\pi$ 的期望 $\mathbb { E } _ { A \sim \pi } [ * ]$。如何做到？直接使用 $\bar { x }$ 是不行的，因为 $\bar {x} \rightarrow \mathbb{E}_{X\sim p_1}[X]$。我们可以通过使用**重要性采样**技术来实现。

## 2. 重要性采样

注意到：
$$
\begin{equation}
\mathbb {E} _ {X \sim p _ {0}} [ X ] = \sum_ {x} p _ {0} (x) x = \sum_ {x} p _ {1} (x) \underbrace {\frac {p _ {0} (x)}{p _ {1} (x)}} _ {f (x)} x = \mathbb {E} _ {X \sim p _ {1}} [ f (X) ]
\end{equation}
$$
因此，我们可以通过估计 $\mathbb { E } _ { X \sim p _ { 1 } } [ f ( X ) ]$ 来估计 $\mathbb { E } _ { X \sim p _ { 0 } } [ X ]$。

如何估计 $\mathbb { E } _ { X \sim p _ { 1 } } [ f ( X ) ]$？很简单。令：
$$
\begin{equation}
\bar {f} \doteq \frac {1}{n} \sum_ {i = 1} ^ {n} f (x _ {i}), \quad \text {where } x _ {i} \sim p _ {1}
\end{equation}
$$
那么
$$
\begin{equation}
\mathbb {E} _ {X \sim p _ {1}} [ \bar {f} ] = \mathbb {E} _ {X \sim p _ {1}} [ f (X) ], \quad \operatorname {v a r} _ {X \sim p _ {1}} [ \bar {f} ] = \frac {1}{n} \operatorname {v a r} _ {X \sim p _ {1}} [ f (X) ]
\end{equation}
$$

因此，$\bar { f }$ 是 $\mathbb { E } _ { X \sim p _ { 0 } } [ X ]$ 的一个很好的近似：
$$
\begin{equation}
\mathbb {E} _ {X \sim p _ {0}} [ X ] \approx \bar {f} = \frac {1}{n} \sum_ {i = 1} ^ {n} f (x _ {i}) = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {p _ {0} (x _ {i})}{p _ {1} (x _ {i})} x _ {i}
\end{equation}
$$
其中：

- $\frac{p_{0}(x_{i})}{p_{1}(x_{i})}$ 被称为**重要性权重**。
- 如果 $p _ { 1 } ( x _ { i } ) = p _ { 0 } ( x _ { i } )$，重要性权重为 1，$\bar { f }$ 就变成了 $\bar { x }$。
- 如果 $p _ { 0 } ( x _ { i } ) \geq p _ { 1 } ( x _ { i } )$，意味着 $x _ { i }$ 在 $p _ { 0 }$ 下比在 $p _ { 1 }$ 下更常被采样到。重要性权重 $( > 1 )$ 可以强调这个样本的重要性。

你可能会问：$\bar{f} = \frac{1}{n} \sum_{i=1}^n \frac{p_0(x_i)}{p_1(x_i)} x_i$ 需要知道 $p_0(x)$，但如果我知道 $p_0(x)$，为什么不直接计算期望呢？

答案是：这种方法适用于以下情况：给定 $x$ 时容易计算 $p_0(x)$，但难以直接计算期望。例如，连续情形，$p_0$ 的表达式复杂，或根本没有显式表达式（例如，$p_0$ 由神经网络表示）。

总结一下，如果 $\{ x _ { i } \} \sim p _ { 1 }$，则有：
$$
\begin{array}{l} \bar {x} = \frac {1}{n} \sum_ {i = 1} ^ {n} x _ {i} \rightarrow \mathbb {E} _ {X \sim p _ {1}} [ X ] \\ \bar {f} = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {p _ {0} \left(x _ {i}\right)}{p _ {1} \left(x _ {i}\right)} x _ {i} \rightarrow \mathbb {E} _ {X \sim p _ {0}} [ X ] \\ \end{array}
$$

## 3. off-policy 策略梯度定理

假设 $\beta$ 是生成经验样本的**行为策略**。我们的目标是使用这些样本来更新一个**目标策略** $\pi$，以最大化度量指标：
$$
\begin{equation}
J (\theta) = \sum_ {s \in \mathcal {S}} d _ {\beta} (s) v _ {\pi} (s) = \mathbb {E} _ {S \sim d _ {\beta}} [ v _ {\pi} (S) ]
\end{equation}
$$
其中 $d _ { \beta }$ 是策略 $\beta$ 下的平稳分布。

在折扣因子 $\gamma \in ( 0 , 1 )$ 的情况下，我们直接给出 $J ( \theta )$ 的梯度为：
$$
\begin{equation}
\nabla_ {\theta} J (\theta) = \mathbb {E} _ {S \sim \rho , A \sim \beta} \left[ \frac {\pi (A | S , \theta)}{\beta (A | S)} \nabla_ {\theta} \ln \pi (A | S, \theta) q _ {\pi} (S, A) \right]
\end{equation}
$$
其中 $\beta$ 是行为策略，$\rho$ 是一个状态分布。$\pi$ 相当于前面例子中的 $p_0$ 分布，$\beta$ 则相当于 $p_1$ 分布。

## 4. off-policy Actor-Critic 算法

离策略策略梯度对于基线 $b ( s )$ 也具有不变性，我们有：
$$
\begin{equation}
\nabla_ {\theta} J (\theta) = \mathbb {E} _ {S \sim \rho , A \sim \beta} \left[ \frac {\pi (A | S , \theta)}{\beta (A | S)} \nabla_ {\theta} \ln \pi (A | S, \theta) \big (q _ {\pi} (S, A) - b (S) \big) \right]
\end{equation}
$$
为了减小估计方差，我们可以选择基线为 $b ( S ) = v _ { \pi } ( S )$，得到：
$$
\begin{equation}
\nabla_ {\theta} J (\theta) = \mathbb {E} \left[ \frac {\pi (A | S , \theta)}{\beta (A | S)} \nabla_ {\theta} \ln \pi (A | S, \theta) \big (q _ {\pi} (S, A) - v _ {\pi} (S) \big) \right]
\end{equation}
$$

相应的随机梯度上升算法为：
$$
\begin{equation}
\theta_ {t + 1} = \theta_ {t} + \alpha_ {\theta} \frac {\pi (a _ {t} | s _ {t} , \theta_ {t})}{\beta (a _ {t} | s _ {t})} \nabla_ {\theta} \ln \pi (a _ {t} | s _ {t}, \theta_ {t}) \left(q _ {t} (s _ {t}, a _ {t}) - v _ {t} (s _ {t})\right)
\end{equation}
$$
与 on-policy 情况类似，可以用 TD 误差近似优势函数：
$$
\begin{equation}
q _ {t} \left(s _ {t}, a _ {t}\right) - v _ {t} \left(s _ {t}\right) \approx r _ {t + 1} + \gamma v _ {t} \left(s _ {t + 1}\right) - v _ {t} \left(s _ {t}\right) \doteq \delta_ {t} \left(s _ {t}, a _ {t}\right)
\end{equation}
$$
于是算法变为：
$$
\begin{equation}
\theta_ {t + 1} = \theta_ {t} + \alpha_ {\theta} \frac {\pi (a _ {t} | s _ {t} , \theta_ {t})}{\beta (a _ {t} | s _ {t})} \nabla_ {\theta} \ln \pi (a _ {t} | s _ {t}, \theta_ {t}) \delta_ {t} (s _ {t}, a _ {t})
\end{equation}
$$
进而可以写成：
$$
\begin{equation}
\theta_ {t + 1} = \theta_ {t} + \alpha_ {\theta} \left(\frac {\delta_ {t} (s _ {t} , a _ {t})}{\beta (a _ {t} | s _ {t})}\right) \nabla_ {\theta} \pi (a _ {t} | s _ {t}, \theta_ {t})
\end{equation}
$$

基于重要性采样的 off-policy Actor-Critic 算法伪代码如下：

$$
\begin{equation*}
\begin{aligned}
& \text{Initialization: A given behavior policy } \beta(a|s). \text{A target policy } \pi(a|s, \theta_0) \text{ where } \theta_0 \text{ is the initial parameter vector.} \\
& \text{A value function } v(s, w_0) \text{ where } w_0 \text{ is the initial parameter vector.} \\
& \text{Aim: Search for an optimal policy maximizing } J(\theta). \\
& \text{At time step } t \text{ in episode, do} \\
& \quad \quad \text{Generate } a_t \text{ following } \beta(s_t), \text{ and then observe } r_{t+1}, s_{t+1}. \\
& \quad \quad \text{TD error (advantage function): } \\
& \quad \quad \quad \quad \delta_ {t} = r _ {t + 1} + \gamma v \left(s _ {t + 1}, w _ {t}\right) - v \left(s _ {t}, w _ {t}\right) \\
& \quad \quad \text{Critic (value update): } \\
& \quad \quad \quad \quad w _ {t + 1} = w _ {t} + \alpha_ {w} \frac {\pi (a _ {t} | s _ {t} , \theta_ {t})}{\beta (a _ {t} | s _ {t})} \delta_ {t} \nabla_ {w} v (s _ {t}, w _ {t}) \\
& \quad \quad \text{Actor (policy update): } \\
& \quad \quad \quad \quad \theta_ {t + 1} = \theta_ {t} + \alpha_ {\theta} \frac {\pi \left(a _ {t} \mid s _ {t} , \theta_ {t}\right)}{\beta \left(a _ {t} \mid s _ {t}\right)} \delta_ {t} \nabla_ {\theta} \ln \pi \left(a _ {t} \mid s _ {t}, \theta_ {t}\right)
\end{aligned}
\end{equation*}
$$

# 四、Deterministic Actor-Critic (DPG)

到目前为止，策略梯度方法中使用的策略都是**随机**的，因为对于每个 $( s , a )$，都有 $\pi ( a | s , \theta ) > 0$。那么我们可以在策略梯度方法中使用**确定性**策略吗？好处：它可以处理**连续动作**空间，这种动作空间有无限个动作。

策略一般表示为 $\pi ( a | s , \theta ) \in [ 0 , 1 ]$，它可以是随机的也可以是确定性的。现在，我们将确定性策略特别表示为：
$$
\begin{equation}
a = \mu (s, \theta) \dot {=} \mu (s)
\end{equation}
$$
其中：

- $\mu$ 是从状态 $s$ 到动作 $\mathcal { A }$ 的一个映射。
- $\mu$ 可以用神经网络表示，输入为 $s$，输出为 $a$，参数为 $\theta$，我们常简写为 $\mu ( s )$。

## 1. 确定性策略梯度定理

之前介绍的策略梯度定理仅适用于随机策略，如果策略必须是确定性的，我们必须推导一个新的策略梯度定理，思路和步骤是类似的。

考虑折扣因子下的平均状态值度量：
$$
\begin{equation}
J (\theta) = \mathbb {E} [ v _ {\mu} (s) ] = \sum_ {s \in \mathcal {S}} d _ {0} (s) v _ {\mu} (s)
\end{equation}
$$
其中 $d _ { 0 } ( s )$ 是一个概率分布，满足 $\sum _ { s \in \mathcal { S } } d _ { 0 } ( s ) = 1$。$d _ { 0 }$ 被选为与 $\mu$ 无关，这样梯度计算更容易。选择 $d _ { 0 }$ 有两种重要且特殊的情况：

- $d _ { 0 } ( s _ { 0 } ) = 1$ 且 $d _ { 0 } ( s \neq s _ { 0 } ) = 0$，其中 $s _ { 0 }$ 是一个特定的起始状态。
- $d _ { 0 }$ 是一个与 $\mu$ 不同的行为策略的平稳分布。

在折扣因子 $\gamma \in ( 0 , 1 )$ 的情况下，我们直接给出 $J ( \theta )$ 的梯度为：
$$
\begin{equation}
\begin{aligned}
\nabla_ {\theta} J (\theta) &= \sum_ {s \in \mathcal {S}} \rho_ {\mu} (s) \nabla_ {\theta} \mu (s) \left(\nabla_ {a} q _ {\mu} (s, a)\right) | _ {a = \mu (s)} \\ 
&= \mathbb {E} _ {S \sim \rho_ {\mu}} \left[ \nabla_ {\theta} \mu (S) {\color{blue}{\left(\nabla_ {a} q _ {\mu} (S, a)\right) | _ {a = \mu (S)}}} \right]
\end{aligned}
\end{equation}
$$
这里，$\rho _ { \mu }$ 是一个状态分布。其中蓝色部分表示 $q_\mu$ 先对 $a$ 求梯度，然后将 $a=\mu(s)$ 代入。与随机情况的一个重要区别是：上式的梯度中**不涉及动作 A 的分布**。因此，确定性策略梯度方法天然是**off-policy**的。

## 2. 确定性 Actor-Critic 算法

基于上述策略梯度，最大化 $J ( \theta )$ 的梯度上升算法为：
$$
\begin{equation}
\theta_ {t + 1} = \theta_ {t} + \alpha_ {\theta} \mathbb {E} _ {S \sim \rho_ {\mu}} \left[ \nabla_ {\theta} \mu (S) \big (\nabla_ {a} q _ {\mu} (S, a) \big) | _ {a = \mu (S)} \right]
\end{equation}
$$
对应的随机梯度上升算法为：
$$
\begin{equation}
\theta_ {t + 1} = \theta_ {t} + \alpha_ {\theta} \nabla_ {\theta} \mu (s _ {t}) (\nabla_ {a} q _ {\mu} (s _ {t}, a)) | _ {a = \mu (s _ {t})}
\end{equation}
$$

确定性 Actor-Critic 算法的伪代码如下：

$$
\begin{equation*}
\begin{aligned}
& \text{Initialization: A given behavior policy } \beta(a|s). \text{A deterministic target policy } \mu(s, \theta_0) \text{ where } \theta_0 \text{ is the initial} \\
& \text{parameter vector. A value function } v(s, w_0) \text{ where } w_0 \text{ is the initial parameter vector.} \\
& \text{Aim: Search for an optimal policy maximizing } J(\theta). \\
& \text{At time step } t \text{ in episode, do} \\
& \quad \quad \text{Generate } a_t \text{ following } \beta(s_t), \text{ and then observe } r_{t+1}, s_{t+1}. \\
& \quad \quad \text{TD error (advantage function): } \\
& \quad \quad \quad \quad \delta_ {t} = r _ {t + 1} + \gamma q \left(s _ {t + 1}, \mu \left(s _ {t + 1}, \theta_ {t}\right), w _ {t}\right) - q \left(s _ {t}, a _ {t}, w _ {t}\right) \\
& \quad \quad \text{Critic (value update): } \\
& \quad \quad \quad \quad w _ {t + 1} = w _ {t} + \alpha_ {w} \delta_ {t} \nabla_ {w} q (s _ {t}, a _ {t}, w _ {t}) \\
& \quad \quad \text{Actor (policy update): } \\
& \quad \quad \quad \quad \theta_ {t + 1} = \theta_ {t} + \alpha_ {\theta} \nabla_ {\theta} \mu \left(s _ {t}, \theta_ {t}\right) \left(\nabla_ {a} q \left(s _ {t}, a, w _ {t + 1}\right)\right) | _ {a = \mu \left(s _ {t}\right)}
\end{aligned}
\end{equation*}
$$

几点说明：

- 这是一个 off-policy 实现，行为策略 $\beta$ 可能与 $\mu$ 不同。
- $\beta$ 也可以用 $\mu$ + 噪声代替。
- 如何选择函数来表示 $q ( s , a , w )$？
  - 线性函数：$q ( s , a , w ) = \phi ^ { T } ( s , a ) w$，其中 $\phi ( s , a )$ 是特征向量。
  - 神经网络：深度确定性策略梯度 (DDPG) 方法。

# 参考

[强化学习的数学原理](https://www.bilibili.com/video/BV1sd4y167NS/?spm_id_from=333.1387.favlist.content.click&vd_source=d3aa59b76ff20c1218700a8386312891)

{% endraw %}
