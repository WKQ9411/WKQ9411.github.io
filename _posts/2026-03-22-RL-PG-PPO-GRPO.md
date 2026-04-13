---
layout: post
title: 【强化学习】从策略梯度到PPO再到GRPO
date: 2026-03-22 16:00:00 +08:00
summary: 从策略梯度（Policy Gradient，PG）到近端策略优化（Proximal Policy Optimization, PPO），再到群组相对策略优化（Group Relative Policy Optimization, GRPO）的逐步介绍。
categories: RL
excerpt_separator: <!--more-->
---

{% raw %}
# 目录
{:.no_toc}

* TOC
{:toc}

---


# 一、强化学习基础回顾

在深入策略梯度之前，我们先简单回顾一下强化学习的一些基本概念：

* **智能体（Agent）**：学习和决策的实体。

* **环境（Environment）**：智能体所处的外部世界。

* **状态（State, $S$）**：环境在某一时刻的描述，$S$为随机变量，具体某个状态为$s$。

* **动作（Action, $A$）**：智能体在某一状态下可以采取的行动，$A$为随机变量，具体某个动作为$a$。

* **策略（Policy, $\pi$）**：智能体从状态到动作的映射，告诉智能体在给定状态下应该采取什么动作。可以是确定性策略：$a = \pi(s)$，或随机性策略：$\pi(a\|s) = P(a\|s)$。

* **奖励（Reward, $R$）**：环境对智能体动作的反馈，$R$为随机变量，具体某个奖励为$r$。

* **轨迹（Trajectory, $\tau$）**：在时刻$t$，智能体处于状态$S\_t$，按照策略$\pi$采取动作$A\_t$，转移到下一个状态$S\_{t+1}$，获得的即时奖励是$R\_{t+1}$（有些地方也写成$R\_t$，不影响理解），这个过程可以表达为：$S\_t \xrightarrow{A\_t} S\_{t+1}, R\_{t+1}$，一系列状态-动作-奖励的序列即形成一个轨迹：$s\_0, a\_0, r\_1, s\_1, a\_1, r\_2, \dots$。

* **回报（Return, $G\_t$）**：从某一时刻$t$开始的所有未来奖励的折扣和：$G\_t \doteq R\_{t+1} + \gamma R\_{t+2} + \gamma^2 R\_{t+3} + \dots = \sum\_{k=t}^{T} \gamma^{k-t} R\_{k+1} = R\_{t+1} + \gamma G\_{t+1}$，其中$\gamma \in (0, 1)$是折扣因子。

* **价值函数（Value Function）**：
    * **状态价值函数（State-Value Function, $V\_\pi(s)$）**：从状态$s$开始，遵循策略$\pi$的期望回报。$V\_\pi(s) = \in \mathbb{E}\_\pi[G\_t \| S\_t = s]$。
    * **动作价值函数（Action-Value Function, $Q\_\pi(s, a)$）**：在状态$s$采取动作$a$后，遵循策略$\pi$的期望回报。$Q\_\pi(s, a) = \mathbb{E}\_\pi[G\_t \| S\_t = s, A\_t = a]$。
    * 状态值和动作值的关系：
    
    $$
    v_{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \left[ \sum_{r \in \mathcal{R}} p(r|s,a) r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) v_{\pi}(s') \right]
    $$
    
    ​	即状态值是该状态对应的动作值的期望，其中:

    $$
    \begin{aligned}
    q_{\pi}(s,a) &= \sum_{r \in \mathcal{R}} p(r|s,a)r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a)v_{\pi}(s') \\
    &= \mathbb{E}\left[ R_{t+1} + \gamma v_\pi(S_{t+1})|S_t=s,A_t=a \right]
    \end{aligned}
    $$

    ​	即动作值是**即时奖励**加上折扣的下一个状态的**状态值**。
    
* **优势函数（Advantage Function, $\delta\_\pi(s, a)$）**：衡量在状态$s$采取动作$a$相对于平均水平（$V\_\pi(s)$）的优势。$\delta\_\pi(s, a) = Q\_\pi(s, a) - V\_\pi(s)$。

强化学习的目标是找到一个最优策略$\pi^\*$，使得智能体在环境中获得的长期回报最大化。

# 二、策略梯度（Policy Gradient，PG）

策略梯度方法直接优化策略，而不是像Q-learning或SARSA那样优化价值函数。它通过一个参数化的策略$\pi\_\theta(a\|s)$（例如，一个神经网络），学习调整参数$\theta$，使得目标函数最大化。用于定义最优策略的目标函数有如下两种：**平均状态值**：$\bar{v}\_\pi=\mathbb{E}\_{S \sim d}[v\_\pi(S)$和**平均奖励**：$\bar{r}\_\pi=\mathbb{E}\_{S \sim d\_\pi}[r\_\pi(S)]$。根据**策略梯度定理**，可以推导出目标函数$J(\theta)$的梯度是（省略推导）：

$$
\begin{equation}
\boxed{ \nabla_{\theta}J(\theta)=\mathbb{E}_{S \sim \eta, A \sim \pi(S, \theta)}\left[\nabla_{\theta} \ln \pi(A | S, \theta) q_{\pi}(S, A)\right]}
\end{equation}
$$

其中，$\eta$是状态的概率分布。得到梯度后，我们就可以通过梯度上升的方法最大化目标函数：

$$
\begin{equation}
\begin{aligned}
\theta_{t + 1} &= \theta_{t}+\alpha\nabla_{\theta}J(\theta_{t}) \\
&= \theta_{t}+\alpha\mathbb{E}\left[\nabla_{\theta} \ln \pi(A|S, \theta_{t})q_{\pi}(S, A)\right]
\end{aligned}
\end{equation}
$$

上式中的真实梯度含有期望，而这在实际中是未知的，根据随机梯度算法，我们可以使用随机梯度替换真实梯度，得到：

$$
\begin{equation}
\theta_{t + 1} = \theta_{t}+\alpha \nabla_{\theta} \ln \pi(a_t|s_t, \theta_{t})q_t(s_t, a_t)
\end{equation}
$$

其中，$q\_t(s\_t,a\_t)$是对$q\_\pi(s\_t,a\_t)$在$t$时刻的估计值。因此，策略参数$\theta\_t$的更新依赖于对动作值的估计$q\_t(s\_t,a\_t)$。

## （一）REINFORCE

如果$q\_t(s\_t,a\_t)$是通过**蒙特卡罗**估计得到的，该算法被称为REINFORCE，其算法流程为：

>**初始化**：初始参数$\theta$；$\gamma \in (0,1)$；$\alpha > 0$。  
>**目标**：学习一个最优策略从而最大化$J(\theta)$。  
>对于每个回合：  
>​&emsp;&emsp;根据$\pi(\theta)$生成$\{s\_0,a\_0,r\_1,\dots,s\_{T-1},a\_{T-1},r\_T\}$  
>​&emsp;&emsp;对于$t=0,1,\dots,T-1$：  
>​&emsp;&emsp;​&emsp;&emsp;价值更新：$q\_t(s\_t,a\_t)=\sum\_{k=t+1}^T \gamma^{k-t-1}r\_k$  
>​&emsp;&emsp;​&emsp;&emsp;策略更新：$\theta \leftarrow \theta + \alpha \nabla\_{\theta} \ln \pi(a\_t\|s\_t, \theta\_{t})q\_t(s\_t, a\_t)$

## （二）Actor-Critic

如果$q\_t(s\_t,a\_t)$是通过**时序差分**方法估计得到的，该算法被称为Actor-Critic。Actor对应的是策略更新，Critic对应的是价值更新，如果Critic使用**Sarsa算法**，这种Actor-Critic算法也被称为Q Actor-Critic（QAC），其算法流程为：

>**初始化**：一个策略函数$\pi(a\|s,\theta\_0)$，其中$\theta\_0$是初始参数。一个价值函数$q(s,a,w\_0)$，其中$w\_0$是初始参数。$\alpha\_w,\alpha\_\theta >0$。  
>**目标**：学习一个最优策略来最大化$J(\theta)$。  
>在每个回合中的$t$时刻：  
>​&emsp;&emsp;根据$\pi(a\|s\_t,\theta\_t)$产生$a\_t$，观测$r\_{t+1},s\_{t+1}$，然后根据$\pi(a\|s\_{t+1},\theta\_t)$生成$a\_{t+1}$  
>​&emsp;&emsp;Actor（策略更新）：  
>​&emsp;&emsp;​&emsp;&emsp;$\theta\_{t+1} = \theta\_t + \alpha\_\theta \nabla\_{\theta} \ln \pi(a\_t\|s\_t, \theta\_{t})q(s\_t, a\_t, w\_t)$  
>​&emsp;&emsp;Critic（价值更新）：  
>​&emsp;&emsp;​&emsp;&emsp;$w\_{t+1} = w\_t + \alpha\_w [r\_{t+1} + \gamma q(s\_{t+1},a\_{t+1},w\_t) - q(s\_t,a\_t,w\_t)] \nabla\_w q(s\_t,a\_t,w\_t)$

## （三）Advantage Actor-Critic（A2C）

策略梯度有一个重要性质：它对额外的基准是不变的，即：

$$
\begin{equation}
\mathbb{E}_{S\sim\eta,A\sim\pi}\left[\nabla_{\theta}\ln\pi(A|S,\theta_t)q_{\pi}(S,A)\right] = \mathbb{E}_{S\sim\eta,A\sim\pi}\left[\nabla_{\theta}\ln\pi(A|S,\theta_t)(q_{\pi}(S,A) - b(S))\right]
\end{equation}
$$

其中，$b(S)$是基准函数，它是$S$的一个标量函数。上式表明添加或去掉一个基准函数$b(S)$不会影响策略梯度。该式成立的原因如下：

$$
\begin{equation}
\begin{aligned}
\mathbb{E}_{S\sim\eta,A\sim\pi}\left[\nabla_{\theta}\ln\pi(A|S,\theta_t)b(S)\right] 
&= \sum_{s\in\mathcal{S}} \eta(s) \sum_{a\in\mathcal{A}} \pi(a|s,\theta_t) \nabla_{\theta}\ln\pi(a|s,\theta_t)b(s) \\
&= \sum_{s\in\mathcal{S}} \eta(s) \sum_{a\in\mathcal{A}} \nabla_{\theta}\pi(a|s,\theta_t)b(s) \\
&= \sum_{s\in\mathcal{S}} \eta(s)b(s) \sum_{a\in\mathcal{A}} \nabla_{\theta}\pi(a|s,\theta_t) \\
&= \sum_{s\in\mathcal{S}} \eta(s)b(s) \nabla_{\theta} \sum_{a\in\mathcal{A}} \pi(a|s,\theta_t) \\
&= \sum_{s\in\mathcal{S}} \eta(s)b(s) \nabla_{\theta} 1 = 0
\end{aligned}
\end{equation}
$$

因此，不会对策略梯度产生影响。实际中，通常使用的基准为$v\_\pi(s)$，此时梯度上升算法变为：

$$
\begin{equation}
\begin{aligned}
\theta_{t + 1} &= \theta_t + \alpha \mathbb{E}\left[\nabla_{\theta}\ln\pi(A|S,\theta_t)[q_{\pi}(S,A) - v_{\pi}(S)]\right] \\
&\doteq \theta_t + \alpha \mathbb{E}\left[\nabla_{\theta}\ln\pi(A|S,\theta_t)\delta_{\pi}(S,A)\right]
\end{aligned}
\end{equation}
$$

其中，$\delta\_{\pi}(S,A)$是**优势函数**，它反映了一个动作相对于其他动作的优势。具体来说，由于状态值$v\_\pi(s)=\sum\_{a \in \mathcal{A}}\pi(a\|s)q\_\pi(s,a)$是平均动作值，因此$\delta\_\pi(s,a)>0$意味着相应的动作值大于均值，具有一定的优势。将上式中的真实梯度换为随机梯度，得到：

$$
\begin{equation}
\boxed{
\begin{aligned}
\theta_{t+1} &= \theta_t + \alpha \nabla_\theta \ln \pi(a_t | s_t, \theta_t) [q_t(s_t, a_t) - v_t(s_t)] \\
&= \theta_t + \alpha \nabla_\theta \ln \pi(a_t | s_t, \theta_t) \delta_t(s_t, a_t)
\end{aligned}
} \label{eq:a2c}
\end{equation}
$$

其中$s\_t,a\_t$是在$t$时刻$S,A$的样本，这里$q\_t(s\_t,a\_t)$和$v\_t(s\_t)$分别是$q\_{\pi(\theta\_t)}(s\_t,a\_t)$和$v\_{\pi(\theta\_t)}(s\_t)$的估计值。

下面进一步对公式$\eqref{eq:a2c}$进行分析。上式中的$\nabla\_\theta\ln\pi$是始终大于0的，因此，若$\delta\_t>0$，则$\nabla\_\theta J(\theta)>0$，目标函数会朝着变大的方向，反之则朝着变小的方向。此外，由于

$$
\nabla_\theta \ln \pi(a_t|s_t,\theta_t) = \frac{\nabla_\theta \pi (a_t|s_t,\theta_t)}{\pi (a_t|s_t,\theta_t)}
$$

公式$\eqref{eq:a2c}$可以重写为

$$
\theta_{t+1} = \theta_{t} + \alpha \underbrace{\left( \frac{\delta_t(s_t, a_t)}{\pi(a_t | s_t, \theta_t)} \right)}_{\beta_t} \nabla_{\theta} \pi(a_t | s_t, \theta_t)
 = \theta_{t} + \alpha {\beta_t} \nabla_{\theta} \pi(a_t | s_t, \theta_t)
$$

当$\theta\_{t+1}-\theta\_t$足够小时，根据一阶泰勒展开可知

$$
\begin{aligned}
\pi(a_t | s_t, \theta_{t+1}) &\approx \pi(a_t | s_t, \theta_t) + (\nabla_{\theta} \pi(a_t | s_t, \theta_t))^T (\theta_{t+1} - \theta_t) \\
&= \pi(a_t | s_t, \theta_t) + \alpha \beta_t (\nabla_{\theta} \pi(a_t | s_t, \theta_t))^T (\nabla_{\theta} \pi(a_t | s_t, \theta_t)) \\
&= \pi(a_t | s_t, \theta_t) + \alpha \beta_t \lVert \nabla_{\theta} \pi(a_t | s_t, \theta_t) \rVert_2^2
\end{aligned}
$$

因此，当$\delta\_t \geq 0$时，$\beta\_t \geq 0$，此时$\pi(a\_t \| s\_t, \theta\_{t+1}) \geq \pi(a\_t \| s\_t, \theta\_t)$，会使得采取$a\_t$动作的概率增大，反之，则概率降低。总的来说，通过增大优势动作的概率，最大化目标函数$J(\theta)$。

A2C的算法流程为：

>**初始化**：策略函数$\pi(a\|s,\theta\_0)$，其中$\theta\_0$是初始参数。价值函数$v(s,w\_0)$，其中$w\_0$是初始参数。$\alpha\_w,\alpha\_\theta>0$。  
>**目标**：学习一个最优策略来最大化$J(\theta)$。  
>在每个回个中的$t$时刻：  
>​&emsp;&emsp;根据$\pi(a\|s\_t,\theta\_t)$生成$a\_t$，然后得到$r\_{t+1},s\_{t+1}$：  
>​&emsp;&emsp;优势函数（时序差分误差）：  
>​&emsp;&emsp;​&emsp;&emsp;$\delta\_t = r\_{t+1} + \gamma v(s\_{t+1},w\_t) - v(s\_t,w\_t)$  
>​&emsp;&emsp;Actor（策略更新）：  
>​&emsp;&emsp;​&emsp;&emsp;$\theta\_{t+1} = \theta\_t + \alpha\_\theta \delta\_t \nabla\_{\theta} \ln \pi(a\_t\|s\_t, \theta\_{t})$  
>​&emsp;&emsp;Critic（价值更新）：  
>​&emsp;&emsp;​&emsp;&emsp;$w\_{t+1} = w\_t + \alpha\_w \delta\_t \nabla\_w v(s\_t,w\_t)$  

## （四）重要性采样

从真实梯度的表达式可以看出：

$$
\begin{equation}
\nabla_{\theta}J(\theta) = \mathbb{E}_{S\sim\eta,A\sim\pi}\left[\nabla_{\theta}\ln\pi(A|S,\theta_t)(q_{\pi}(S,A) - v_{\pi}(S))\right]
\end{equation}
$$

为了使用随机梯度来近似这个真实梯度，我们必须按照$\pi(\theta)$生成动作样本。因此，$\pi(\theta)$是行为策略，而$\pi(\theta)$也是我们要改进的目标策略，所以策略梯度方法是On-Policy的。如果我们已经有一些由其他行为策略生成的样本，那么策略梯度方法仍然可以使用这些样本来得到最优策略，此时的方法就变成了Off-Policy，不过此时需要采用一种称为重要性采样（Importance Sampling，IS)的技术。

考虑一个随机变量$X \in \mathcal{X}$。假设$p\_0(X)$是一个概率分布，我们的目标是估计$\mathbb{E}\_{X \sim p\_0}[X]$，假设我们有一些独立同分布的样本$\{ x\_i \}\_{i=1}^n$。

1.   样本$\{ x\_i \}\_{i=1}^n$是根据$p\_0$生成的。此时平均值$\bar{x}=\frac{1}{n}\sum\_{i=1}^nx\_i$可以用来近似$\mathbb{E}\_{X \sim p\_0}[X]$，这是因为$\bar{x}$是$\mathbb{E}\_{X \sim p\_0}[X]$的无偏估计。
2.   样本$\{ x\_i \}\_{i=1}^n$不是根据$p\_0$生成的，而是根据另一个概率分布$p\_1$生成的。我们不能再使用平均值$\bar{x}=\frac{1}{n}\sum\_{i=1}^nx\_i$来近似$\mathbb{E}\_{X \sim p\_0}[X]$，这是因为$\bar{x}\approx\mathbb{E}\_{X \sim p\_1}[X]$而非$\mathbb{E}\_{X \sim p\_0}[X]$。

此时我们就需要使用重要性采样技术，具体来说，$\mathbb{E}\_{X \sim p\_0}[X]$满足下式：

$$
\begin{equation}
\begin{aligned}
\mathbb{E}_{X \sim p_0}[X] &= \sum_{x \in \mathcal{X}} p_0(x)x 
= \sum_{x \in \mathcal{X}} p_1(x) \underbrace{\frac{p_0(x)}{p_1(x)}x}_{f(x)}
= \mathbb{E}_{X \sim p_1}[f(X)] \\
&\approx \bar{f} = \frac{1}{n} \sum_{i=1}^n f(x_i) = \frac{1}{n} \sum_{i=1}^n \underbrace{\frac{p_0(x_i)}{p_1(x_i)}}_{\text{重要性权重}} x_i
\end{aligned}
\end{equation}
$$

上式表明，估计$\mathbb{E}\_{X \sim p\_0}[X]$被转换为估计$\mathbb{E}\_{X \sim p\_1}[f(X)]$的问题。当$p\_0(x\_i) \geq p\_1(x\_i)$时，这意味着$x\_i$可以在$p\_0$下更频繁的被采样到，而在$p\_1$下较少的被采样到。此时重要性权重大于1，突出了这个样本的重要性。

利用重要性采样，我们可以推导出Off-Policy策略梯度定理。假设$\beta$是一个行为策略，我们的目标是使用由$\beta$生成的样本来得到一个目标策略$\pi$，从而最大化下面的目标函数：

$$
\begin{equation}
J(\theta)=\sum_{s\in\mathcal{S}}d_{\beta}(s)v_{\pi}(s)=\mathbb{E}_{S\sim d_{\beta}}[v_{\pi}(S)]
\end{equation}
$$

其中，$d\_\beta$是在策略$\beta$下的平稳分布。Off-Policy的策略梯度定理为：

$$
\begin{equation}
\nabla_{\theta}J(\theta)=\mathbb{E}_{S\sim\rho,A\sim\beta}\left[ \underbrace{\frac{\pi(A|S,\theta)}{\beta(A|S)}}_{\text{重要性权重}}\nabla_{\theta}\ln\pi(A|S,\theta)\delta_{\pi}(S,A) \right]
\end{equation}
$$

进一步，有

$$
\begin{equation}
\begin{aligned}
\nabla_{\theta}J(\theta)&=\mathbb{E}_{S\sim\rho,A\sim\beta}\left[ \frac{\pi(A|S,\theta)}{\beta(A|S)} \frac{\nabla_\theta \pi(A|S,\theta)}{\pi(A|S,\theta)} \delta_{\pi}(S,A) \right] \\
&= \mathbb{E}_{S\sim\rho,A\sim\beta}\left[ \frac{\nabla_\theta \pi(A|S,\theta)}{\beta(A|S)}\delta_{\pi}(S,A) \right]
\end{aligned}
\end{equation}
$$

因此目标函数可以写为：

$$
\begin{equation}
\boxed{J(\theta) = \mathbb{E}_{S\sim\rho,A\sim\beta}\left[ \frac{\pi(A|S,\theta)}{\beta(A|S)}\delta_{\pi}(S,A) \right]} \label{eq:offpolicy}
\end{equation}
$$

## （五）Trust Region Policy Optimization（TRPO）

标准的策略梯度方法对步长（学习率）非常敏感。如果更新过大，策略可能会发生剧烈变化，可能进入策略空间的“不良”区域，难以恢复，这会导致训练不稳定和性能不佳 。TRPO，即信任区域策略优化，是一种旨在稳定地改进策略的强化学习算法。其核心思想是在每次迭代中，通过优化一个“替代”目标函数（Surrogate Objective）来小幅更新策略，同时确保新的策略与旧的策略不会偏离太远。这种“偏离程度”由 KL 散度来衡量，从而将策略更新限制在一个“信任区域”内，以期望实现单调的策略性能提升。

TRPO的优化目标函数是一个**带有KL散度约束的替代目标函数**，其数学形式如下：

$$
\begin{equation}
\begin{aligned}
& \underset{\theta}{\text{maximize}} & & \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} \hat{A}_t \right] \\
& \text{subject to} & & \hat{\mathbb{E}}_t[\text{KL}[\pi_{\theta_{\text{old}}}(\cdot | s_t), \pi_\theta(\cdot | s_t)]] \leq \delta
\end{aligned} \label{eq:trpo}
\end{equation}
$$

其中（这里的符号延用了[PPO论文](https://arxiv.org/pdf/1707.06347)中的）：

-   $\pi\_\theta(a\_t\|s\_t)$是新策略的动作概率；
-   $\pi\_{\theta\_{\text{old}}}(a\_t\|s\_t)$是旧策略的动作概率；
-   $\hat{A}\_t$是**基于旧策略**计算的优势函数，衡量动作$a\_t$相对于平均表现的优劣；
-   $\text{KL}(⋅,⋅)$是KL散度，用于约束新旧策略之间的差异不超过阈值$\delta$。

这里需说明的是，公式$\eqref{eq:offpolicy}$和公式$\eqref{eq:trpo}$的形式比较相似，都是在优势函数前乘了一个重要性采样权重，但二者仍有差异。即公式$\eqref{eq:offpolicy}$的优势函数是基于**目标策略**的，也就是**重要性权重的分子**对应的那个策略，而公式$\eqref{eq:trpo}$的优势函数是基于**旧策略**的，也就是**重要性权重的分母**对应的那个策略。

1.   对于Off-Policy策略梯度的公式$\eqref{eq:offpolicy}$，它的目标是学习一个**目标策略**$\pi$的价值，而数据是由**行为策略**$\beta$生成的。优势函数$\delta$为了正确地更新目标策略，需要知道在状态$s$下采取动作$a$对于**目标策略**$\pi$来说有多好。如果优势函数是基于$\beta$的，那么实际上是在评估和改进行为策略，而不是目标策略。【重要性采样（IS）用于调整从$\beta$采样的回报或优势的分布，使其看起来像是从$\pi$采样的】。
2.   TRPO的主要目标是在当前策略$\pi\_{\theta\_{\text{old}}}$的基础上进行稳定且可靠的改进，以获得新策略$\pi\_\theta$。它通过限制新旧策略之间的变化幅度来实现这一点。在一个训练迭代开始时，数据是由当前策略$\pi\_\theta$收集的。优势函数是衡量在执行$\pi\_{\theta\_{\text{old}}}$时所采取行动的相对价值 。这些算法旨在优化一个在$\pi\_{\theta\_{\text{old}}}$“附近”（信任区域内）的策略$\pi\_\theta$。因此，评估动作好坏的基准自然是$\pi\_{\theta\_{\text{old}}}$的表现。我们是在问：“【鉴于这个动作在旧策略下是好/坏的，我们应该如何调整它在新策略中的概率，同时确保新策略不会偏离旧策略太远？】”

总结一下：Off-Policy策略梯度的目标是学习一个可能与行为策略完全不同的**最终目标策略**，因此，优势的评估必须针对这个最终目标策略。TRPO的目标是基于当前（旧）策略进行**迭代改进**，产生一个略微不同但更好的新策略。因此，优势的评估是针对产生数据的那个旧策略的，重要性采样则用于将这些基于旧策略的评估“转移”到新策略的优化上，同时通过KL约束来限制策略变化。TRPO本质上是一种On-Policy算法，因为它用于生成训练数据的策略（行为策略）与正在被评估和改进的策略（目标策略）是相同的。

# 三、近端策略优化（Proximal Policy Optimization, PPO）

PPO旨在改进TRPO的计算效率，同时保持策略更新的稳定性。PPO的核心思想是**限制策略更新的幅度**，避免训练过程中策略的剧烈变化导致性能崩溃。PPO的主要改进在于**用更简单的方式实现策略更新的约束**，而不是像TRPO那样使用复杂的二阶优化方法（如共轭梯度法）。PPO有两种主要变体：

1.  **PPO-Clip（Clipped Surrogate Objective）**：通过裁剪重要性权重来限制策略更新。
2.  **PPO-Penalty（Adaptive KL Penalty）**：在目标函数中加入KL散度惩罚项。

## （一）Clipped Surrogate Objective：PPO-Clip

令$r\_t(\theta)$表示概率比率：$r\_t(\theta) = \frac{\pi\_\theta(a\_t \| s\_t)}{\pi\_{\theta\_{\text{old}}}(a\_t \| s\_t)}$，因此，$r(\theta\_{\text{old}}) = 1$。TRPO最大化以下替代目标函数：

$$
\begin{equation}
L^{CPI}(\theta) = \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} \hat{A}_t \right] = \hat{\mathbb{E}}_t \left[ r_t(\theta) \hat{A}_t \right]
\end{equation}
$$

其中，CPI是指Conservative Policy Iteration，即保守策略迭代。没有约束时，最大化$L^{CPI}$可能会导致过大的策略更新，因此需要考虑修改这一目标函数。为了惩罚策略改变过大，防止$r\_t(\theta)$远离1，PPO-Clip的目标函数修改如下：

$$
\begin{equation}
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t) \right]
\end{equation}
$$

其中：

* $r\_t(\theta) = \frac{\pi\_\theta(a\_t\|s\_t)}{\pi\_{\theta\_{old}}(a\_t\|s\_t)}$ 是重要性采样比率。
* $\hat{A}\_t$ 是优势函数估计（通常使用GAE - Generalized Advantage Estimation）。
* $\epsilon$ 是一个小的超参数，通常取0.1或0.2，用于控制裁剪的范围。
* $\text{clip}(x, 1 - \epsilon, 1 + \epsilon)$表示将$x$限制在$[1 - \epsilon, 1 + \epsilon]$之间。

这个目标函数包含两部分：

-   第一部分：$r\_t(\theta) \hat{A}\_t$，这是未裁剪的重要性采样目标。
-   第二部分：$\text{clip}(r\_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}\_t$，这是裁剪后的目标。

PPO目标函数取这两部分的最小值。这意味着：

-   如果优势函数$\hat{A}\_t>0$（说明当前动作是好的），PPO会鼓励增加该动作的概率。但是，如果$r\_t(\theta)$变得太大（新策略与旧策略偏离太多），它会被裁剪到$1+\epsilon$，从而防止策略更新过于激进。
-   如果优势函数$\hat{A}\_t<0$（说明当前动作是坏的），PPO会鼓励减小该动作的概率。但是，如果$r\_t(\theta)$变得太小（新策略大幅度减小了该动作的概率），它会被裁剪到$1-\epsilon$，防止策略更新过快地抛弃“坏”动作。

<div align="center"><img src="https://wkqpicture.oss-cn-beijing.aliyuncs.com/img/20250528184746334.png" width="70%" alt="image-20250528184738679"></div>

因此，PPO-Clip在$\hat{A}\_t$为正时，会抑制$r\_t(\theta)$过大；在$\hat{A}\_t$为负时，会抑制$r\_t(\theta)$过小。这确保了策略更新不会偏离旧策略太远。

## （二）Adaptive KL Penalty Coefficient：PPO-Penalty

PPO的另一种形式是在目标函数中加入KL散度惩罚项，以限制新旧策略的距离，被称为PPO-Penalty。虽然PPO-Clip更为流行和常用，但PPO-Penalty也是原始PPO论文中提出的一种方法，并且在某些情况下也很有用。

PPO-Penalty的核心思想是将KL散度（Kullback-Leibler divergence）作为**一个惩罚项**加入到策略目标函数中，以限制新策略与旧策略之间的差异。同时，为了更好地控制这种差异，**惩罚系数是自适应调整的**。它的目标函数通常可以表示为：

$$
\begin{equation}
L^{KL PEN}(\theta) = \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} \hat{A}_t - \beta \, \text{KL}[\pi_{\theta_{\text{old}}}(\cdot | s_t), \pi_\theta(\cdot | s_t)] \right]
\end{equation}
$$

自适应KL惩罚系数的目标是使每次策略更新后，新策略与旧策略之间的平均KL散度保持在一个**目标值**（$d\_{\text{targ}}$）附近。具体调整步骤如下：

在每个策略更新周期（即每次使用一批数据进行多次梯度更新）之后：

1.  计算平均KL散度：计算当前策略$\pi\_\theta$与旧策略$\pi\_{\theta\_{\text{old}}}$之间在所有采样数据上的平均KL散度，记为$d = \hat{\mathbb{E}}\_t[\text{KL}[\pi\_{\theta\_{\text{old}}}(\cdot \| s\_t), \pi\_\theta(\cdot \| s\_t)]]$。
2.  调整惩罚系数$\beta$：
    -   如果$d$远大于目标KL散度$d\_{\text{targ}}$（例如，$d>1.5\times d\_{\text{targ}}$），说明策略更新步子迈得太大了，新策略与旧策略偏离太多。此时，增大$\beta$（例如，$\beta \leftarrow 2\beta$），以施加更大的惩罚，迫使未来的策略更新更保守。
    -   如果$d$远小于目标KL散度$d\_{\text{targ}}$（例如，$d<d\_{\text{targ}}/1.5$），说明策略更新步子迈得太小了，没有充分利用学习潜力。此时，减小$\beta$（例如，$\beta \leftarrow \beta/2$），以减小惩罚，允许未来的策略更新更大胆。
    -   如果$d$在目标KL散度$d\_{\text{targ}}$附近（例如，$d\_{\text{darg}} /1.5 \leq d \leq 1.5 \times d\_{\text{darg}}$），说明策略更新的幅度比较合适，此时$\beta$保持不变。

通过这种方式，KL惩罚系数$\beta$会在训练过程中动态地调整，从而确保策略更新的步长既不会太大导致不稳定，也不会太小导致学习效率低下。

此外，PPO还加入了**价值函数损失**（VF）和**熵正则化**（S），最终的目标函数如下：

$$
\begin{equation}
L^{CLIP+VF+S}_t(\theta,\phi) = \hat{\mathbb{E}}_t \left[ L^{CLIP}_t(\theta) - c_1 L^{VF}_t(\phi) + c_2 S[\pi_\theta](s_t) \right]
\end{equation}
$$

其中$c\_1,c\_2$是系数。

1.   PPO通常会同时训练一个价值网络，价值网络的损失函数通常是均方误差：

$$
\begin{equation}
L^{VF}_t(\phi)=(V_\theta(s_t) - V^{\text{targ}}_t)^2
\end{equation}
$$

2.   增加熵正则化项的逻辑是：当策略的熵较高时，智能体选择不同动作的概率分布更均匀，这意味着它会尝试更多不同的动作，从而增加发现潜在高回报动作或路径的机会。如果策略的熵较低，智能体可能会过于自信地只选择它认为最好的动作，从而错过更好的、尚未探索的动作或区域。因此，这一项会鼓励探索，有助于防止策略过早地收敛到局部最优解，提高训练的稳定性。

# 四、群组相对策略优化（Group Relative Policy Optimization, GRPO）

GRPO是DeepSeek在PPO基础上提出的改进算法，旨在提升LLM强化学习微调的效率和稳定性。PPO通常使用一个独立的价值网络（Critic）来估计状态价值。GRPO放弃了价值网络，而是从组分数中估计基线，显著减少了训练资源。

## （一）从PPO到GRPO

首先，再次写出PPO的替代目标函数（这次按照[GRPO论文](https://arxiv.org/pdf/2402.03300)的符号表达）：

$$
\begin{equation}
\begin{aligned}
& J_{PPO}(\theta) = \\
& \mathbb{E}_{q \sim P(Q), o \sim \pi_{\theta_{\text{old}}}(O|q)} \frac{1}{|o|} \sum_{t=1}^{|o|} \min \left[ \frac{\pi_\theta(o_t | q, o_{<t})}{\pi_{\theta_{\text{old}}}(o_t | q, o_{<t})} A_t, \text{clip} \left( \frac{\pi_\theta(o_t | q, o_{<t})}{\pi_{\theta_{\text{old}}}(o_t | q, o_{<t})}, 1 - \epsilon, 1 + \epsilon \right) A_t \right]
\end{aligned}
\end{equation}
$$

其中，$q,o$分别是从问题数据集中采样得到的问题以及从旧策略$\pi\_{\theta\_{\text{old}}}$中采样得到的输出，$q$相当于状态，而$o$相当于执行的动作。$\epsilon$是PPO中引入的与裁剪相关的超参数，用于稳定训练。$A\_t$是优势函数，通过应用广义优势估计（Generalized Advantage Estimation，GAE）来计算。GAE的主要目标是在优势函数的估计中，有效地平衡偏差和方差。

---

这里介绍一下GAE。在估计优势函数时，我们面临一个经典的**偏差-方差权衡（Bias-Variance Trade-off）**问题：

1.   高方差，低偏差（Monte Carlo 方法）：
     -   优势函数的蒙特卡洛估计：我们可以通过让智能体完成一个完整的轨迹，然后计算每个时间步的蒙特卡洛回报 $G\_t = \sum\_{k=0}^{\infty} \gamma^k R\_{t+k+1}$。那么优势函数可以估计为 $\hat{A}\_t = G\_t - V(s\_t)$。
     -   特点：这种方法是无偏的，因为它直接使用实际观察到的回报。但它具有高方差，高方差意味着即使在同一状态下，由于后续轨迹的随机性，每次估计的优势值可能会有很大差异，导致策略更新不稳定。

2.   低方差，高偏差（单步TD误差）：
     -   单步时序差分（TD）误差：这是最简单的优势函数估计之一，即 $\delta\_t = R\_{t+1} + \gamma V(s\_{t+1}) - V(s\_t)$。
     -   特点：这种方法具有低方差，因为它只依赖于一步的奖励和对下一个状态的价值估计，从而减少了未来步骤的随机性。但它具有高偏差，因为 $V(s\_{t+1})$ 本身就是一个估计值，如果价值模型不准确，就会引入偏差。

GAE旨在提供一个更灵活的方法，通过引入一个参数$\lambda (\lambda \in [0, 1])$来权衡这两者，从而得到一个既有较低方差又不会引入过大偏差的优势估计。

GAE的核心思想是将多步TD误差以指数加权平均的形式结合起来。首先，我们定义单步TD误差：

$$
\begin{equation}
\delta_t = R_{t+1} + \gamma V(s_{t+1}) - V(s_t)
\end{equation}
$$

这个 $\delta\_t$ 本身就可以看作是对优势函数的一个单步估计。然后，GAE将不同步长的TD误差进行加权求和。GAE的最终形式通常表示为：

$$
\begin{equation}
\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
\end{equation}
$$

其中：

-   $\gamma$是折扣因子，衡量未来奖励的重要性。

-   $\lambda$是GAE参数，控制偏差和方差之间的权衡。

    -   当$\lambda = 0$时，GAE退化为单步TD误差：

        $$
        \hat{A}_t^{GAE(\gamma, 0)} = \delta_t = R_{t+1} + \gamma V(s_{t+1}) - V(s_t)
        $$

        这种情况下，优势估计方差最低，但偏差最高（因为它完全依赖于价值函数的估计）。

    -   当$\lambda = 1$时，GAE退化为蒙特卡洛优势估计（在价值函数准确的情况下）：

        $$
        \hat{A}_t^{GAE(\gamma, 1)} = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l} = \left( \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \right) - V(s_t)
        $$

        这相当于使用了从当前时间步到轨迹结束的所有未来奖励来估计回报，然后减去当前状态的价值。这种情况下，优势估计是无偏的（如果$V(s)$是真实的），但方差最高。

    -   对于$0 < \lambda < 1$，GAE是介于两者之间的一种估计。它通过指数衰减的方式，给较近的TD误差赋予更高的权重，而给较远的TD误差赋予较低的权重。这使得GAE能够在一定程度上利用多步信息来降低方差，同时又不会像纯蒙特卡洛那样引入极高的方差。

GAE在实际实现中，通常通过从轨迹的末尾向前倒推的方式进行计算，这更高效：

1.   首先，对于轨迹中的最后一个时间步$T$，其优势值通常设为0。

2.   然后，从$t = T - 1$开始，向前计算每个时间步的优势：

     $$
     \hat{A}_t = \delta_t + \gamma \lambda \hat{A}_{t+1}
     $$

     其中$\delta\_t = R\_{t+1} + \gamma V(s\_{t+1}) - V(s\_t)$。这种递归形式的计算非常方便，而且只依赖于当前的$\delta\_t$和下一个时间步的优势 $\hat{A}\_{t+1}$。

---

回到正文。在PPO中，需要与策略模型同时训练一个价值函数，并且为了减轻奖励模型的过度优化，标准方法是在每个token的奖励中添加来自参考模型的每个token的KL散度惩罚，即：

$$
\begin{equation}
r_t = r_\varphi(q, o_{\leq t}) - \beta \log \frac{\pi_\theta(o_t | q, o_{<t})}{\pi_{\text{ref}}(o_t | q, o_{<t})}
\end{equation}
$$

其中，$r\_\varphi$是奖励模型；$\pi\_{\text{ref}}$是参考模型，通常是初始的SFT模型；$\beta$是KL散度的惩罚系数。

由于PPO中使用的价值函数通常是与策略模型规模相当的另一个模型，这带来了巨大的内存和计算负担。此外，在强化学习训练过程中，价值函数在优势函数的计算中被用作基线以减少方差。奖励模型通常只给整个序列的**最后一个 token**一个奖励分数。这意味着很难为序列中的每个中间token分配准确的奖励信号来训练一个精确的价值函数。这种稀疏的奖励信号使得价值函数的训练变得复杂。



在大语言模型的情境中，通常奖励模型仅给最后一个token分配奖励分数，这可能会让在每个token上都精确的价值函数训练变得复杂。为解决此问题，如图4所示，我们提出了组相对策略优化（GRPO）方法。该方法无需像近端策略优化（PPO）那样进行额外的价值函数近似，而是将针对同一问题生成的多个采样输出的平均奖励作为基线。

为了解决上述问题，提出GRPO，消除了对额外价值函数近似的需求，不再需要价值网络。它使用**“the average reward of multiple sampled outputs in response to the same question, as the baseline”** (同一问题下多个采样输出的平均奖励作为基线)，这是GRPO优势计算的关键所在。其具体流程是：对于每个输入问题$q$，它会从当前的（旧的）策略$\pi\_{\theta\_{\text{old}}}$中采样生成一组G个不同的输出$\{o\_1,o\_2,\dots,o\_G\}$，然后最大化以下目标函数：

$$
\begin{equation}
\begin{aligned}
& J_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q)} \\
& \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min \left[ \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})} \hat{A}_{i,t}, \text{clip} \left( \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})}, 1 - \epsilon, 1 + \epsilon \right) \hat{A}_{i,t} \right] - \beta \mathbb{D}_{KL} \left[ \pi_\theta || \pi_{\text{ref}} \right] \right\}
\end{aligned}
\end{equation}
$$

其中，$\hat{A}\_{i,t}$仅仅基于组内输出的**相对奖励**计算而来，将在下文介绍。GRPO的组相对优势计算方法与奖励模型的训练方式天然契合。奖励模型通常通过学习对同一问题不同回答的**比较**（哪个更好，哪个更差）来训练，而不是为每个回答提供绝对的价值评分。此外，PPO 通常是将KL惩罚从奖励中减去，然后用这个修改后的奖励去计算GAE。而GRPO则是**直接将KL散度项添加到总的损失函数中**（作为负值），这可以避免使$\hat{A}\_{i,t}$的计算复杂化。这意味着GRPO的$\hat{A}\_{i,t}$纯粹是基于奖励模型的原始评分进行组内相对计算的，而KL惩罚是作为额外的正则化项直接作用于策略优化。与传统PPO论文不同，GRPO用的是一个**无偏估计器**来计算KL散度：

$$
\begin{equation}
\mathbb{D}_{KL} \left[ \pi_\theta || \pi_{\text{ref}} \right] = \frac{\pi_{\text{ref}}(o_{i,t} | q, o_{i,<t})}{\pi_\theta(o_{i,t} | q, o_{i,<t})} - \log \frac{\pi_{\text{ref}}(o_{i,t} | q, o_{i,<t})}{\pi_\theta(o_{i,t} | q, o_{i,<t})} - 1
\end{equation}
$$

其值总是非负的。

此外，需注意目标函数中涉及到的两个平均：

-   **组平均**：$\frac{1}{G} \sum\_{i=1}^{G}$，表示对组内所有G个输出的贡献进行平均。

-   **序列平均**：$\frac{1}{\|o\_i\|} \sum\_{t=1}^{\|o\_i\|}$，表示对每个输出序列中的所有token进行平均。这意味着GRPO不仅仅关注最终奖励，而是对每个token的生成概率进行优化。

PPO与GRPO的区别如下图所示：

<div align="center"><img src="https://wkqpicture.oss-cn-beijing.aliyuncs.com/img/20250528210552612.png" width="100%" alt="image-20250528210552038"></div>

## （二）奖励监督机制

奖励监督机制有两种：**Outcome Supervision RL (结果监督强化学习)** 和 **Process Supervision RL (过程监督强化学习)**。这两种方法的核心区别在于奖励信号是在整个输出序列的**末尾**给出，还是在生成过程的**每一步**给出。

**结果监督**

1.   采样输出组：对于每个问题$q$，从旧策略模型$\pi\_{\theta\_{old}}$中采样生成一组$G$个不同的输出：$\{o\_1, o\_2, \dots, o\_G\}$
2.   奖励评分：一个奖励模型被用来对这$G$个输出分别进行评分，得到对应的$G$个奖励值：$\mathbf{r} = \{r\_1, r\_2, \dots, r\_G\}$
3.   奖励归一化：这些奖励会进行归一化处理。归一化方法是减去组的平均值，并除以组的标准差。
     -   标准化后的奖励：$\tilde{r}\_i = \frac{r\_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$

4.   优势计算：将这个标准化后的奖励$\tilde{r}\_i$设置为该输出$o\_i$中所有token的优势$\hat{A}\_{i,t}$。也就是说，对于输出$o\_i$中的任何一个token $t$，它的优势都是相同的，即$\hat{A}\_{i,t} = \tilde{r}\_i$。
5.   策略优化：然后，通过最大化GRPO目标函数来优化策略。

结果监督的奖励信号只在整个输出序列的末尾（即生成了完整的$o\_i$后）才给出。实现相对简单，适用于那些只有最终结果才有明确奖励的场景（如判断某个回答是否正确）。然而，对于复杂的、多步骤的推理任务，这种稀疏的奖励信号可能不足以有效地指导模型学习每一步的正确决策，因为模型无法得知中间步骤哪里做得好，哪里做得不好。

**过程监督**

1.   采样输出组：同样，对于每个问题$q$，采样一组$G$个输出$\{o\_1, o\_2, \dots, o\_G\}$
2.   过程奖励：这里使用一个过程奖励模型（Process Reward Model）。这个模型被设计用来评分输出的每一步，而不仅仅是最终结果。产生的奖励为：$\mathbf{R} = \{\{r\_1^{\text{index}(1)}, \cdots, r\_1^{\text{index}(K\_1)}\}, \cdots, \{r\_G^{\text{index}(1)}, \cdots, r\_G^{\text{index}(K\_G)}\}\}$，其中，$\text{index}(j)$代表第$j$步结束的token索引，$K\_i$ 是第 $i$ 个输出的总步骤数。
3.   分步奖励归一化：与结果监督类似，这些分步奖励也会进行归一化，得到$\tilde{r}\_i^{\text{index}(j)} = \frac{r\_i^{\text{index}(j)} - \text{mean}(\mathbf{R})}{\text{std}(\mathbf{R})}$
4.   优势计算（累积求和）：过程监督的优势$\hat{A}\_{i,t}$不再是单个标准化奖励，而是从当前token $t$到序列末尾的所有后续标准化过程奖励的和，即$\hat{A}\_{i,t} = \sum\_{j \geq \text{index}(t)} \tilde{r}\_i^{\text{index}(j)}$。这意味着，一个token的优势不仅取决于它所在步骤的奖励，还取决于其之后所有步骤的奖励。这是一个类似于“未来回报”的累积概念。
5.   策略优化：最后，同样通过最大化GRPO目标函数来优化策略。

## （三）迭代式强化学习

随着策略模型（Policy Model）变得越来越强大和智能，它可能会开始生成奖励模型（Reward Model）在训练时从未见过的新颖（或更复杂、更精妙）的输出。如果奖励模型仍然停留在“旧”的水平，它可能无法准确地评估这些新颖输出的真实质量，或者无法区分细微的性能差异。这会导致奖励信号变得不准确或不足够，从而限制策略模型的进一步学习和优化。这就像一个总是用旧考试标准来评判新时代学生表现的老师，最终会阻碍学生的进步。

在GRPO的迭代过程中，策略模型会生成新的输出。这些新的、由当前策略模型生成的输出，会被用来构建**新的训练集**，专门用于更新奖励模型。新生成的数据会和10%的历史数据结合起来训练奖励模型。这有助于奖励模型学习最新的输出模式，以更好地评估当前策略模型生成的内容。通过包含历史数据，避免遗忘对早期、更简单输出的评估能力，保持其泛化性和鲁棒性。

此外，在每个迭代周期开始时，将当前训练中的策略模型的参数复制并冻结，作为新的参考模型（Reference Model）。这种做法确保了KL散度惩罚项始终是相对于一个“最近”的、性能较好的基准来计算的，而不是一个停滞不前的旧模型。然后，策略模型会继续使用这个最新训练好的奖励模型来提供奖励信号，并根据这些信号进行优化。

最终，GRPO的算法流程为：

>**初始化**：策略模型$\pi\_{\theta\_{\text{init}}}$；奖励模型$r\_\varphi$；任务提示词$\mathcal{D}$；超参数$\epsilon, \beta, \mu$。  
>策略模型$\pi\_\theta \leftarrow \pi\_{\theta\_{\text{init}}}$  
>每个迭代`iteration = 1, ..., I`执行：  
>​&emsp;&emsp;参考模型$\pi\_{\text{ref}} \leftarrow \pi\_\theta$  
>​&emsp;&emsp;每个时间步`step = 1, ..., M`执行：  
>​&emsp;&emsp;​&emsp;&emsp;从 $\mathcal{D}$ 中采样一个batch得到$\mathcal{D}\_b$  
>​&emsp;&emsp;​&emsp;&emsp;更新旧策略模型$\pi\_{\theta\_{\text{old}}} \leftarrow \pi\_\theta$  
>​&emsp;&emsp;​&emsp;&emsp;对每个$q \in \mathcal{D}\_b$，采$G$个输出$\{o\_i\}\_{i=1}^G \sim \pi\_{\theta\_{\text{old}}}(\cdot \| q)$  
>​&emsp;&emsp;​&emsp;&emsp;通过奖励模型$r\_\varphi$计算每个输出$o\_i$的奖励$\{r\_i\}\_{i=1}^G$  
>​&emsp;&emsp;​&emsp;&emsp;通过组相对优势估计计算$o\_i$第$t$个token的优势$\hat{A}\_{i,t}$  
>​&emsp;&emsp;​&emsp;&emsp;每个GRPO迭代`iteration = 1, ..., μ`执行：  
>​&emsp;&emsp;​&emsp;&emsp;​&emsp;&emsp;通过最大化GRPO目标函数更新策略模型$\pi\_\theta$  
>​&emsp;&emsp;​&emsp;&emsp;通过使用回放机制（replay mechanism）持续训练更新奖励模型$r\_\varphi$
{% endraw %}
