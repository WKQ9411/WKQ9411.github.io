---
layout: post
title: 【论文解读】DeepSeek-R1
date: 2026-01-01 16:00:00 +08:00
summary: 解读DeepSeek-R1论文。
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

[DeepSeek-R1](https://arxiv.org/abs/2501.12948)进行了强化学习在提升模型推理能力方面的探索，其内容主要包括三个部分：

-   DeepSeek-R1-Zero：直接应用强化学习（RL）于基础模型，无需监督微调（SFT）
-   DeepSeek-R1：包含两个强化学习阶段和两个监督微调阶段
-   Distilled Model：大型模型的推理模式成功蒸馏至小型模型

# 一、DeepSeek-R1-Zero：在 Base Model 上直接进行 RL

DeepSeek-R1-Zero 探索了大型语言模型在没有任何监督数据的情况下发展推理能力的潜力，重点关注通过纯粹的强化学习过程进行自我演化。在训练流程上，DeepSeek-R1-Zero 很简单，使用基于规则的 Reasoning Data 进行纯强化学习（图片引自：[DeepSeek-R1 论文解读及相关技术杂谈](https://pfcc.blog/posts/swagger-deepseek-r1)）：

<div align="center"><img src="https://wkqpicture.oss-cn-beijing.aliyuncs.com/img/20250601193431409.png" width="70%" alt="image-20250601193431346"></div>

## （一）强化学习算法

其使用的强化学习方法是**群组相对策略优化（Group Relative Policy Optimization，GRPO）**，详细的介绍参见[【笔记】从策略梯度到PPO再到GRPO](/2026-01-01/Note-PG-PPO-GRPO.html)。

## （二）奖励模型

奖励决定了 RL 的优化方向，为了训练 DeepSeek-R1-Zero，使用了基于规则的奖励系统，主要包括两种类型的奖励：

-   准确性奖励（Accuracy Rewards）：准确性奖励模型用于评估响应是否正确。例如，对于具有确定性结果的数学问题，模型被要求以指定格式提供最终答案，从而实现可靠的基于规则的正确性验证。类似地，对于 LeetCode 问题，可以使用编译器针对预定义的测试用例生成反馈。
-   格式奖励（Format Rewards）：除了准确性奖励模型外，还采用了一种格式奖励模型，强制模型将其思考过程置于`<think>`和`</think>`标签之间。

在开发 DeepSeek-R1-Zero 时，没有应用基于结果或过程的神经奖励模型（outcome or process neural reward model），因为在大规模强化学习过程中，神经奖励模型可能会遭受奖励黑客（reward hacking）的困扰，并且重新训练奖励模型需要额外的训练资源，还会使整个训练流程复杂化。神经奖励模型通常是另一个神经网络，它通过学习数据来判断模型输出的好坏。奖励黑客是指：在大规模强化学习中，模型可能会找到一些“取巧”的方法来获得高奖励，但这些方法并没有真正解决问题或达到预期的目标。神经奖励模型由于其复杂性和学习特性，可能更容易被模型“欺骗”或利用其漏洞。

## （三）数据构造

为了训练 DeepSeek-R1-Zero，首先设计了简洁的模板，用于指导 base model 遵循指定的指令。如下图所示：

<div align="center"><img src="https://wkqpicture.oss-cn-beijing.aliyuncs.com/img/20250601175017187.png" width="100%" alt="image-20250601175010023"></div>

模板的约束主要在于输出的“结构”（即先推理、后答案），而不是对推理“内容”本身做过多限制。特意避免在模板中引入可能导致偏见的内容要求。例如，不强制模型必须使用“反思性推理”（reflective reasoning，一种更深入、可能包含自我校正的推理方式），也不刻意引导模型采用某一种特定的“解决问题策略”。这样做的目的是为了能够在强化学习过程中，更真实、更准确地观察和评估模型自身是如何学习和发展其解决问题能力的。如果模板对推理内容干预过多，研究者可能就无法分清哪些能力的提升是模型自主学习的结果，哪些仅仅是对模板指令的机械模仿。通过给予模型在推理内容上的更大自由度，他们希望看到模型在强化学习信号的引导下“自然地”进化和进步。

## （四）DeepSeek-R1-Zero 的性能、自我进化过程和 Aha Moment

### 1. 性能

下图展示了 DeepSeekR1-Zero 经过 RL 的性能，可见其达到了与 OpenAI-o1-0912 相当的性能水平。这一显著改进突显了 RL 算法在优化模型性能方面的有效性。

<div align="center"><img src="https://wkqpicture.oss-cn-beijing.aliyuncs.com/img/20250601190719709.png" width="80%" alt="image-20250601190719473"></div>

### 2. 自我进化过程

随着训练的深入，模型的思考时间逐渐增加。这一进步并非源于外部调整，而是模型自我发展的结果。DeepSeek-R1-Zero 通过延长测试时间，自然地提升了解决复杂推理任务的能力。自我进化的一个显著特征是，随着测试时间和计算量的增加，**复杂行为自然涌现**。例如，模型会自发反思，重新审视和评估先前的步骤，并探索解决问题的替代方法。这些行为并非通过显式编程实现，而是模型与强化学习环境交互的结果。

<div align="center"><img src="https://wkqpicture.oss-cn-beijing.aliyuncs.com/img/20250601191328307.png" width="80%" alt="image-20250601191328130"></div>

### 3. Aha Moment

"Aha Moment"即“顿悟时刻”，模型能够在推理过程中能发现自己的错误，并继续思考纠正这个错误，这是以往 SFT 没有出现的能力。按照论文，训练过程中并没有使用包含自我纠错格式的数据，因此"Aha Moment"的出现属于数据分布外的现象，证明了强化学习能够带来意想不到的推理能力提升。

<div align="center"><img src="https://wkqpicture.oss-cn-beijing.aliyuncs.com/img/20250601192359549.png" width="80%" alt="image-20250601192359375"></div>

>This moment is not only an “aha moment” for the model but also for the researchers observing its behavior.  
>可以看得出来团队在写这段文字的时候也是感到激动人心的。

尽管 DeepSeek-R1-Zero 展现出强大的推理能力，并能自主发展出出乎意料且强大的推理行为，但它仍面临一些问题。例如，DeepSeek-R1-Zero 在处理诸如可读性差、语言混杂等挑战时存在困难。

# 二、DeepSeek-R1：使用冷启动的 RL

受 DeepSeek-R1-Zero 的启发，引出两个问题：

1.   能否通过引入少量高质量数据作为冷启动来进一步提升推理性能或加速收敛？
2.   如何训练一个用户友好的模型，该模型不仅能够生成清晰连贯的思维链（Chain of Thought, CoT），还能展现出强大的泛化能力？

为解决这些问题，论文提出了训练 DeepSeek-R1 的流程，该流程包含以下四个阶段：

-   Stage1：冷启动微调（Cold Start SFT）
-   Stage2：面向推理的强化学习（Reasoning-oriented Reinforcement Learning）
-   Stage3：拒绝采样和监督微调（Rejection Sampling and Supervised Fine-Tuning）
-   Stage4：全场景强化学习（Reinforcement Learning for all Scenarios）

其实这可以分成两部分来看，**DeepSeek-V3 Base** 经过 Stage1 和 Stage2 得到中间模型，这个模型是为了生成更多高质量的 Reasoning Data；接着 **DeepSeek-V3 Base** 经过 Stage3+Stage4 得到最终的 **DeepSeek-R1**。Step1 和 Step3 都是 SFT 阶段，其实都是 **Policy Initialization** 阶段，都是为了让模型在进行强化学习前有更好的基础能力（指令跟随、 推理能力）。其流程如下（引自：[DeepSeek-R1 论文解读及相关技术杂谈](https://pfcc.blog/posts/swagger-deepseek-r1)）：

<div align="center"><img src="https://wkqpicture.oss-cn-beijing.aliyuncs.com/img/20250601194529629.png" width="100%" alt="image-20250601194529481"></div>

## （一）冷启动

与 DeepSeek-R1-Zero 不同，为了防止基础模型在 RL 训练早期出现不稳定的冷启动阶段，研究人员为 DeepSeek-R1 构建并收集了少量**长思维链（long CoT）数据**来微调基础模型，使其作为初始的 RL actor。“冷启动阶段”指的是在强化学习训练的最初期，基础模型对于特定任务的理解和执行能力还很弱，其行为可能接近随机，导致训练过程不稳定，学习效率低下。通过这个微调步骤，模型在正式开始强化学习之前，就已经具备了一定的按照思维链进行推理和解决问题的能力，从而有了一个更好的起点，使得后续的强化学习过程更加稳定和高效。

使用了以下几种方法来收集长思维链数据：

-   **使用包含长思维链示例的少样本提示 (using few-shot prompting with a long CoT as an example)：** “少样本提示”是指在给模型下达指令（prompt）时，提供几个（few-shot）输入输出的范例。在这里，这些范例本身就是“长思维链”的例子，模型通过学习这些例子来生成类似的、包含详细推理过程的输出。
-   **直接提示模型生成包含反思和验证的详细答案 (directly prompting models to generate detailed answers with reflection and verification)：** 通过精心设计的提示语，直接要求模型不仅给出答案，还要在其输出中包含“反思”和“验证”的过程。这种方式旨在生成更高质量、更可靠的思维链数据。
-   **收集 DeepSeek-R1-Zero 以可读格式输出的结果 (gathering DeepSeek-R1-Zero outputs in a readable format)：** 利用 DeepSeek-R1-Zero 的输出，将这些输出整理成更“可读”、更适合作为训练数据的格式。
-   **通过人工标注员进行后处理来优化结果 (refining the results through post-processing by human annotators)：** 无论数据是通过哪种方式生成的，最后都会由人工标注员进行审查和“后处理”，进一步提升这些长思维链数据的质量。

在这一步，收集了数千条冷启动数据来微调 DeepSeek-V3-Base，作为 RL 的起点。与 DeepSeek-R1-Zero 相比，冷启动数据的优势包括：

-   **Readability（可读性）**：DeepSeek-R1-Zero 的一个主要局限性是其内容通常不适合阅读，回复中可能混合多种语言，或缺少用于为用户高亮答案的 Markdown 格式。相比之下，在为 DeepSeek-R1 创建冷启动数据时，设计了一种易阅读的模式，在每个回复的末尾包含一个摘要，并过滤掉那些不方便阅读的回复。即将输出格式定义为：`|special_token |<reasoning_process>|special_token |<summary>`，其中`<reasoning_process>`是针对问题的思维链（CoT），而`<summary>`则用于总结推理结果。
-   **Potential（潜力）**：通过精心设计带有人类先验知识的冷启动数据模式，可以观察到其性能优于 DeepSeek-R1-Zero。研究人员相信，**迭代训练**是推理模型更好的一种训练方式。

## （二）面向推理的强化学习

在冷启动数据上对 DeepSeek-V3-Base 进行微调之后，**采用了与 DeepSeek-R1-Zero 中相同的规模化强化学习训练过程**。此阶段专注于增强模型的推理能力，尤其是针对诸如编码、数学、科学和逻辑推理等推理密集型任务，这些任务通常包含定义明确的问题和清晰的答案。

在训练过程中，可以观察到思维链（CoT）经常表现出语言混合现象，尤其是当强化学习的提示涉及多种语言时。为了缓解语言混合问题，研究人员在强化学习训练期间引入了一种**语言一致性奖励**，该奖励根据思维链中目标语言词语的比例来计算。尽管消融实验表明这种语言一致性对齐会导致模型性能略有下降，但这种奖励符合人类偏好，使其更具可读性。最后，通过将推理任务的准确性奖励和语言一致性奖励直接相加来组合形成最终奖励。然后，对微调后的模型进行强化学习训练，直到它在推理任务上达到收敛为止。

## （三）拒绝采样和监督微调

当上一阶段（面向推理的强化学习）收敛后，利用其产生的检查点来收集用于下一轮的 SFT 数据。与主要侧重推理的初始冷启动数据不同，这一阶段整合了来自其他领域的数据，以提升模型在写作、角色扮演及其他通用任务方面的能力。具体来说，按照下述方式生成数据并微调模型：

-   **Reasoning data（推理数据）**：
    -   精心设计或挑选“推理提示”（reasoning prompts），然后用上一阶段 RL 训练好的模型 checkpoint 来针对这些提示通过**拒绝采样**的方法生成“推理轨迹”（reasoning trajectories）。
    -   在上一阶段，只包含了那些可以使用基于规则的奖励进行评估的数据。而在这个阶段，进一步整合额外的数据来扩展数据集，其中一些数据使用了生成式奖励模型，即通过将真实标签和模型预测输入到 DeepSeek-V3 中实现数据质量的评判。
    -   此外，由于模型输出有时比较混乱且难以阅读，过滤掉了那些语言混合、段落过长以及包含代码块的思维链。对于每个提示，采样多个回复，并只保留正确的那些。
    -   总共收集了大约 **60 万条**与推理相关的训练样本。
-   **Non-Reasoning data（非推理数据）**：
    -   对于非推理数据，例如写作、事实性问答、自我认知和翻译，采用了 DeepSeek-V3 的 pipeline 并复用 DeepSeek-V3 SFT 数据集的部分内容。
    -   对于某些非推理任务，通过提示词调用 DeepSeek-V3，在回答问题前生成一个潜在的思维链。然而，对于更简单的查询，例如“你好”，则不在回复中提供思维链。
    -   总共大约 **20 万条**与推理无关的训练样本。

使用上述样本对 DeepSeek-V3-Base 进行了 2 个 epoch 的 SFT 微调。

## （四）全场景强化学习

为了进一步使模型与人类偏好对齐，实施了一个次级强化学习阶段，旨在提高模型的有用性（helpfulness）和无害性（harmlessness），同时继续优化其推理能力。具体来说，使用**奖励信号组合**和**多样化的提示分布**来训练模型。

-   **对于推理数据**，遵循 DeepSeek-R1-Zero 中概述的方法论，利用**基于规则**的奖励来指导在数学、代码和逻辑推理领域的学习过程。
-   **对于通用数据**，依靠**奖励模型**来捕捉复杂和细微场景中的人类偏好。基于 DeepSeek-V3 的 pipeline，采用类似的偏好对和训练提示词分布。
-   **在有用性方面**，主要关注最终的摘要，评估标准是摘要内容对用户是否真的有用、是否切题，同时最大限度地减少对底层推理过程的干扰。
-   **在无害性方面**，评估模型的整个回复，包括推理过程和摘要，以识别和减轻在生成过程中可能出现的任何潜在风险、偏见或有害内容。

最终，通过奖励信号和多样化数据分布的整合，训练出一个在推理方面表现出色，同时兼具有用性和无害性的模型。

# 三、Distilled Model：赋予小模型推理能力

为了让更高效的小型模型具备类似 DeepSeek-R1 的推理能力，使用由 DeepSeek-R1 精心筛选的 80 万条样本直接微调了像 Qwen 和 Llama 这样的开源模型。研究结果表明，这种直接的蒸馏方法显著增强了小型模型的推理能力。这里使用的基础模型是 Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, Qwen2.5-14B, Qwen2.5-32B, Llama-3.1-8B, 和 Llama-3.3-70B-Instruct。其流程如下（引自：[DeepSeek-R1 论文解读及相关技术杂谈](https://pfcc.blog/posts/swagger-deepseek-r1)）：

<div align="center"><img src="https://wkqpicture.oss-cn-beijing.aliyuncs.com/img/20250601232332462.png" width="80%" alt="image-20250601232332382"></div>

对于蒸馏的模型，仅应用了监督微调（SFT），并未包含强化学习（RL），尽管整合强化学习可以显著提升模型性能。这里主要目标是展示蒸馏技术的有效性。

# 四、结尾

关于模型评估部分略去，几个结论值得关注：

1.   对于小型模型，**蒸馏方法比直接应用强化学习更有效**。
2.   虽然蒸馏策略既经济又有效，但超越智能边界的推进可能仍需要更强大的基础模型和更大规模的强化学习。

# 五、参考链接

1.   [DeepSeek-R1 论文解读及相关技术杂谈](https://pfcc.blog/posts/swagger-deepseek-r1)
{% endraw %}
