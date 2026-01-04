---
layout: post
title: 【笔记】MLA矩阵吸收分析
date: 2026-01-01 16:00:00 +08:00
summary: 详细分析MLA中的矩阵吸收计算方式。
categories: Note
---

{% raw %}
# 目录
{:.no_toc}

* TOC
{:toc}

---

# 一、张量运算的计算量

## 1. FLOPs定义

**FLOPs**：Floating Point Operations 指的是浮点运算次数，一般特指乘加运算次数，**理解为计算量**，可以用来衡量算法/模型时间的复杂度。更大的计算量单位通常包括：

- **MFLOPs**：百万次浮点运算（$10^6$ FLOPs）。
- **GFLOPs**：十亿次浮点运算（$10^9$ FLOPs）。
- **TFLOPs**：万亿次浮点运算（$10^{12}$ FLOPs）。

张量运算的计算量通常与运算维度和操作类型有关，以`pytorch`中线性层`nn.Linear`的计算为例，设输入张量的维度为$B \times S \times D$，线性层内部权重矩阵维度为$D \times O$：

-   若不考虑`bias`，两个张量相乘的结果维度为$B \times S \times O$，结果中的每个元素是由原始张量分别沿着$D$维度**进行了$D$次乘法和$D-1$次加法**而来的，因此总计算量为：

$$
(2D-1)\times B \times S \times O
$$

-   若考虑`bias`，则每个元素由原始张量分别沿着$D$维度**进行$D$次乘法和$D-1$次加法后，还需加上`bias`，因此一共也执行了$D$次加法**，总计算量为：

$$
2D \times B \times S \times O
$$

为了简单起见，后续分析时均以考虑`bias`来分析，这样FLOPs的计算可直接由相关维度的相乘而来。

## 2. 张量计算顺序对计算量的影响

张量计算顺序的不同会影响计算量。以下是一个例子：

假设有三个张量 $A$、$B$ 和 $C$，它们的形状分别为：
- $A$: $(m, n)$
- $B$: $(n, p)$
- $C$: $(p, q)$

我们需要计算 $A \times B \times C$，其中 $\times$ 表示矩阵乘法。

**计算顺序 1**：先计算 $A \times B$，再乘以 $C$

1. 计算 $A \times B$：
   - 结果形状为 $(m, p)$。
   - 每个元素的计算量为 $2n$（$n$ 次乘法和 $n$ 次加法）。
   - **总计算量**：$m \times p \times 2n = 2mnp$。
2. 计算 $(A \times B) \times C$：
   - 结果形状为 $(m, q)$。
   - 每个元素的计算量为 $2p$（$p$ 次乘法和 $p$ 次加法）。
   - **总计算量**：$m \times q \times 2p = 2mpq$。
3. **总计算量**：$2mnp + 2mpq$。

**计算顺序 2**：先计算 $B \times C$，再乘以 $A$

1. 计算 $B \times C$：
   - 结果形状为 $(n, q)$。
   - 每个元素的计算量为 $2p$（$p$ 次乘法和 $p$ 次加法）。
   - **总计算量**：$n \times q \times 2p = 2npq$。
2. 计算 $A \times (B \times C)$：
   - 结果形状为 $(m, q)$。
   - 每个元素的计算量为 $2n$（$n$ 次乘法和 $n$ 次加法）。
   - **总计算量**：$m \times q \times 2n = 2mnq$。
3. **总计算量**：$2npq + 2mnq$。

**比较两种计算顺序**：

- **计算顺序 1**的总计算量为 $2mnp + 2mpq$。

- **计算顺序 2**的总计算量为 $2npq + 2mnq$。

- 将上述两式相减，有：

    $$
    2[mn(p-q)+pq(m-n)]
    $$

    可见如果$p<q,m<n$则必定计算顺序1的计算量更小，如果$p>q,m>n$则反之，其余情况 则需根据具体数值分析。

# 二、MLA第一次矩阵吸收的计算量分析

我们比较三种计算顺序：

假设原始序列$\mathbf{h}$经Q低秩压缩后得到$\mathbf{c}^Q$，经KV低秩压缩得到$\mathbf{c}^{KV}$，它们的上投影矩阵分别为$W^{UQ}$和$W^{UK}$。

## 1. 原始注意力计算

原始注意力计算如下：

$$
(W^{UQ}\mathbf{c}^Q)^T (W^{UK}\mathbf{c}^{KV})
$$

上述张量的形状如下，箭头右边是简记的符号，并将`n_heads × qk_nope_head_dim`进行了拆分：

- $W^{UQ}$ ：`(q_lora_rank, n_heads × qk_nope_head_dim) -> (q, h, d)`
- $\mathbf{c}^Q$ ：`(bsz, q_seq_len, q_lora_rank) -> (b, s, q)`
- $W^{UK}$ ：`(kv_lora_rank, n_heads × qk_nope_head_dim) -> (k, h, d)`
- $\mathbf{c}^{KV}$ ：`(bsz, k_seq_len, kv_lora_rank) -> (b, t, k)`
- Step 1： $W^{UQ}\mathbf{c}^Q$：`(bsz, q_seq_len, n_heads, qk_nope_head_dim) -> (b, s, h, d)`
- Step 2：$W^{UK}\mathbf{c}^{KV}$：`(bsz, k_seq_len, n_heads, qk_nope_head_dim) -> (b, t, h, d)`
- Step 3：$(W^{UQ}\mathbf{c}^Q)^T (W^{UK}\mathbf{c}^{KV})$：`(bsz, n_heads, q_seq_len, k_seq_len) -> (b, h, s, t)`

这里**区分q_seq_len和k_seq_len**，训练或prefill时二者是一致的，decode时q_seq_len是1，k_seq_len是cache的长度。

根据张量计算量分析的规则，计算量如下：

$$
\text{FLOPs}_{\text{order}_1}=2bshdq+2bthdk+2bhstd
$$

## 2. MLA源代码中的吸收方式

$$
[(W^{UQ}\mathbf{c}^Q)^T W^{UK}]\mathbf{c}^{KV}
$$

-   Step 1：$W^{UQ}\mathbf{c}^Q$：`(bsz, q_seq_len, n_heads, qk_nope_head_dim) -> (b, s, h, d)`
-   Step 2：$(W^{UQ}\mathbf{c}^Q)^TW^{UK}$：`(bsz, q_seq_len, n_heads, kv_lora_rank) -> (b, s, h, k)`
-   Step 3：$[(W^{UQ}\mathbf{c}^Q)^T W^{UK}]\mathbf{c}^{KV}$：`(bsz, n_heads, q_seq_len, k_seq_len) -> (b, h, s, t)`

计算量如下：

$$
\text{FLOPs}_{\text{order}_2}=2bshdq+2bshkd+2bhstk
$$

## 3. 提前吸收

$$
{\mathbf{c}^Q}^T(W^{UQ^T} W^{UK})\mathbf{c}^{KV}
$$

-   Step 1：$W^{UQ^T} W^{UK}$：`(n_heads, q_lora_rank, kv_lora_rank) -> (h, q, k)`
-   Step 2：${\mathbf{c}^Q}^T(W^{UQ^T} W^{UK})$：`(bsz, q_seq_len, n_heads, kv_lora_rank) -> (b, s, h, k)`
-   Step 3：${\mathbf{c}^Q}^T(W^{UQ^T} W^{UK})\mathbf{c}^{KV}$：`(bsz, n_heads, q_seq_len, k_seq_len) -> (b, h, s, t)`

计算量如下：

$$
\text{FLOPs}_{\text{order}_3}=2hqkd+2bshkq+2bhstk
$$

## 4. 比较分析

### 4.1 比较顺序1和顺序2

首先比较$\text{FLOPs}_{\text{order}_1}$和$\text{FLOPs}_{\text{order}_2}$，有：

$$
\text{FLOPs}_{\text{order}_1}-\text{FLOPs}_{\text{order}_2}=
2bhdk(t-s)+2bhst(d-k)
$$

其中：

-   `t`：`k_seq_len`
-   `s`：`q_seq_len`
-   `d`：`qk_nope_head_dim = 128`
-   `k`：`kv_lora_rank = 512`
-   `h`：`n_heads = 128`
-   `b`：`bsz`由于第一项和第二项都有`b`，为简单起见，设为1

在训练或prefill阶段，`t`=`s`，上式结果为$-98304s^2$，此时顺序1的计算量更优。

在decode阶段，`t`是缓存长度，而`s`=1，上式结果为$16777216(t-1)-98304t=16678912t-16777216$，可见，推理时随着缓存长度`t`的变大，顺序1需要花费更大的计算量，因此才需要把$W^{UK}$吸收进$W^{UQ}\mathbf{c}^Q$（也就是代码中的`q_nope`）中，避免产生的中间量需要大量的计算。

### 4.2 比较顺序2和顺序3

然后比较$\text{FLOPs}_{\text{order}_2}$和$\text{FLOPs}_{\text{order}_3}$，有：

$$
\text{FLOPs}_{\text{order}_2}-\text{FLOPs}_{\text{order}_3}=
2hdq(bs-k)+2bshk(d-q)
$$

其中：

-   `q`：`q_lora_rank = 1536`
-   `b`：`bsz`第一项的`b`无法作为因子提出，因此先不假定具体值

上式结果中不包含`t`，结果为$50331648(bs-512)-184549376bs=-134217728bs-25769803776$，恒小于0，因此顺序2的计算量优于顺序3。其原因是$(W^{UQ^T} W^{UK})$充当了新的$W^{UQ'}$，其形状为`(h, q, k)`，具有100663296个元素。而$W^{UQ}$和$W^{UK}$的形状分别为`(q, h, d)`和`(k, h, d)`，二者之和只有33554432个元素，约为$W^{UQ'}$的33%，这就解释了虽然公式上直接将$W^{UK}$吸收进了$W^{UQ}$，但为什么代码实现上不这么做的原因。不论是从参数量占用还是计算量上，顺序3都没有优势。

# 三、MLA第二次矩阵吸收的计算量分析

同样比较三种计算顺序：

假设得到的`score`形状大小为`(bsz, n_heads, q_seq_len, k_seq_len)`，$\mathbf{c}^{KV}$向`value`的上投影矩阵为$W^{UV}$，输出维度变换 矩阵为$W^O$。

## 1. 原始输出计算

原始计算顺序如下：

$$
W^O[score(W^{UV} \mathbf{c}^{KV})]
$$

上述张量的形状如下，将`n_heads × v_head_dim`进行了拆分：

-   $score$：`(bsz, n_heads, q_seq_len, k_seq_len) -> (b, h, s, t)`
-   $\mathbf{c}^{KV}$：`(bsz, k_seq_len, kv_lora_rank) -> (b, t, k)`
-   $W^{UV}$：`(kv_lora_rank, n_heads × v_head_dim) -> (k, h, v)`
-   $W^O$：`(n_heads × v_head_dim, dim) -> (h, v, e)`
-   Step 1：$W^{UV} \mathbf{c}^{KV}$：`(bsz, k_seq_len, n_heads, v_head_dim) -> (b, t, h, v)`
-   Step 2：$[score(W^{UV} \mathbf{c}^{KV})]$：`(bsz, n_heads, q_seq_len, v_head_dim) -> (b, h, s, v)`
-   Step 3：$W^O[score(W^{UV} \mathbf{c}^{KV})]$：`(bsz, n_heads, q_seq_len, dim) -> (b, h, s, e)`

计算量如下：

$$
\text{FLOPs}_{\text{order}_1}=2bthvk+2bhsvt+2bhsev
$$

## 2. MLA源代码中的吸收方式

$$
W^O[W^{UV} (score\mathbf{c}^{KV})]
$$

- Step 1：$score\mathbf{c}^{KV}$：`(bsz, n_heads, q_seq_len, kv_lora_rank) -> (b, h, s, k)`
- Step 2：$[W^{UV} (score\mathbf{c}^{KV})]$：`(bsz, n_heads, q_seq_len, v_head_dim) -> (b, h, s, v)`
- Step 3：$W^O[W^{UV} (score\mathbf{c}^{KV})]$：`(bsz, n_heads, q_seq_len, dim) -> (b, h, s, e)`

计算量如下：

$$
\text{FLOPs}_{\text{order}_2}=2bhskt+2bhsvk+2bhsev
$$

## 3. 提前吸收

$$
(W^OW^{UV})(score\mathbf{c}^{KV})
$$

- Step 1：$W^OW^{UV}$：`(n_heads, kv_lora_rank, dim) -> (h, k, e)`
- Step 2：$score\mathbf{c}^{KV}$：`(bsz, n_heads, q_seq_len, kv_lora_rank) -> (b, h, s, k)`
- Step 3：$(W^OW^{UV})(score\mathbf{c}^{KV})$：`(bsz, n_heads, q_seq_len, dim) -> (b, h, s, e)`

计算量如下：

$$
\text{FLOPs}_{\text{order}_3}=2hkev+2bhskt+2bhsek
$$

## 4. 比较分析
### 4.1 比较顺序1和顺序2

首先比较$\text{FLOPs}_{\text{order}_1}$和$\text{FLOPs}_{\text{order}_2}$，有：

$$
\text{FLOPs}_{\text{order}_1}-\text{FLOPs}_{\text{order}_2}=2bhvk(t-s)+2bhst(v-k)
$$

其中：
-   `t`：`k_seq_len`
-   `s`：`q_seq_len`
-   `v`：`v_head_dim = 128`
-   `k`：`kv_lora_rank = 512`
-   `h`：`n_heads = 128`
-   `b`：`bsz`由于第一项和第二项都有`b`，为简单起见，设为1

由于`v`与`d`值大小一样，因此计算结果与与第一次矩阵吸收一致。即在训练或prefill阶段，顺序1更优，在decode阶段，顺序2更优。

### 4.2 比较顺序2和顺序3

然后比较$\text{FLOPs}_{\text{order}_2}$和$\text{FLOPs}_{\text{order}_3}$，有：

$$
\text{FLOPs}_{\text{order}_2}-\text{FLOPs}_{\text{order}_3}=2hvk(bs-e)+2bhse(v-k)
$$

其中：
- `e`：`dim = 7168`
- `b`：`bsz`第一项的`b`无法作为因子提出，因此先不假定具体值

上式结果为$16777216(bs-7168)-704643072bs=-687865856bs −120259084288$，可见仍然是顺序2的计算结果更优。
# 参考链接

1.   [训练模型算力的单位：FLOPs、FLOPS、Macs 与 估算模型（FC, CNN, LSTM, Transformers&&LLM）的FLOPs - 知乎](https://zhuanlan.zhihu.com/p/649993943)
2.   [llm 参数量-计算量-显存占用分析 - Zhang](https://www.armcvai.cn/2024-09-20/llm-params-flops.html#二-计算量分析)
3.   [DeepSeek-V3 MLA 优化全攻略：从低秩压缩到权重吸收，揭秘高性能推理的优化之道 - 知乎](https://zhuanlan.zhihu.com/p/25449691772)
{% endraw %}
