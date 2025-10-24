---
layout: post
title: "SFT under the scanner"
date: 2025-08-09 00:00:00 -0000
author: Santanu Pattanayak
tags: UAT, Universal approximation theorem, Neural networks, 
---

# Table of Contents


## Introduction <a name="introduction"></a>

Supervised Fine-Tuning (SFT) has long been the first post-training stage for large language models. After pretraining on massive unlabeled corpora, models are refined using curated instruction–response pairs to follow human-like directions and adhere to factual, stylistic, or ethical norms. This process is simple and stable but reveals key limitations.

SFT minimizes a cross-entropy loss that forces imitation of target responses. While it improves immediate helpfulness and fluency, it often collapses the model’s output distribution around demonstrated behaviors—diminishing diversity, creativity, and even factual robustness in open-ended settings.

Reinforcement Learning (RL)–based post-training, especially via Proximal Policy Optimization (PPO), treats the model as a policy that learns from feedback while preserving prior knowledge. Instead of copying demonstrations, PPO aligns the model through reward signals reflecting human or AI preferences, enhancing user satisfaction without distorting its pretrained distribution.

This shift from imitation-driven SFT to alignment-focused RL represents a key evolution in fine-tuning. RL better preserves generalization, prevents distribution collapse, and achieves more stable alignment—even when supervision is sparse. Still, SFT remains an essential precursor, instilling basic instruction-following behaviors that set the stage for effective RL-based alignment.



## The Modern Training Pipeline — Roles of Pretraining, SFT, and Alignment


| **Attribute** | **Pretraining** | **Supervised Fine-Tuning (SFT)** | **Alignment (RLHF / PPO)** |
|----------------|-----------------|----------------------------------|-----------------------------|
| **Primary Objective** | Predict the next token in a sequence — learn general patterns of language, reasoning, and world knowledge by minimizing the difference between predicted and actual text. | Learn to imitate high-quality human or curated responses by minimizing cross-entropy loss over instruction–response pairs. | Optimize model behavior using reward signals from human or AI feedback, improving preference alignment while constraining divergence from the reference policy. |
| **Data Source** | Massive unlabeled text corpora (web, books, academic papers, code, etc.). | Curated instruction–response datasets (human-written or synthetically generated). | Preference data — human or AI rankings, comparisons, or reward model outputs. |
| **Goal / Outcome** | Acquire broad linguistic, factual, and structural understanding of the world. | Teach the model to follow instructions, stay factual, and maintain a coherent conversational style. | Refine the model’s responses to align with human intent, tone, and ethical expectations while preserving its pretrained diversity. |
| **Optimization Method** | Gradient descent on the language modeling objective (e.g., AdamW). | Supervised gradient descent minimizing imitation loss. | Policy-gradient optimization (e.g., PPO or related methods) with reward shaping and KL regularization. |
| **Key Challenges** | Expensive training; data bias; lack of grounding to human values or context. | Distribution collapse around demonstrated examples; loss of diversity and generalization. | Reward hacking; optimization instability; tuning balance between reward gain and knowledge retention. |


Each stage contributes uniquely to the model’s final behavior. **Pretraining** builds a foundation of general knowledge, **SFT** organizes this knowledge into structured instruction-following behavior, and **RL-based alignment** fine-tunes those behaviors to better reflect human preferences without eroding the model’s prior understanding.


## Mathematical Explanation as to why SFT forgets more than RL


While we have been discussing surface-level intuition for why SFT tends to forget more than RL, let’s examine it in greater detail mathematically through their objectives.

In **Supervised Fine-Tuning (SFT)**, the model is trained on curated **instruction–response pairs** $$\{x, y\}$$ from a dataset $$D$$.  
The training objective of SFT is to maximize the **likelihood** of the response $$y$$ conditioned on the instruction $$x$$ under the model.  
If the model is denoted as a policy $$\pi_{\theta}$$, then the loss and its gradient take the following form:

$$
\begin{align}
L_{SFT} &= -\mathbb{E}_{x,y \sim D} [\log \pi_{\theta}(y|x)] \\
\nabla_{\theta} L_{SFT} &= -\mathbb{E}_{x,y \sim D} [\nabla_{\theta}\log \pi_{\theta}(y|x)]
\end{align}
$$

This is equivalent to minimizing the **cross-entropy loss** between the model’s predicted response tokens and the ground-truth tokens, given the prompt $$x$$.  

In **Reinforcement Learning (RL)**, instead of imitating curated instruction–response pairs, the policy $$\pi_{\theta}$$ is updated based on **rewards** $$r(x, y)$$ received for responses $$y$$ sampled from the current policy given a prompt $$x$$.  
For policy-gradient methods, the objective and its gradient are given by:

$$
\begin{align}
L_{RL} &= -\mathbb{E}_{x \sim D} \mathbb{E}_{y \sim \pi_{\theta}(.|x)} [r(x,y)] \\
\nabla_{\theta} L_{RL} &= -\mathbb{E}_{x \sim D} \mathbb{E}_{y \sim \pi_{\theta}(.|x)}[\nabla_{\theta}\log \pi_{\theta}(y|x)r(x,y)]
\end{align}
$$

Comparing $$\nabla_{\theta} L_{SFT}$$ and $$\nabla_{\theta} L_{RL}$$, we can observe two major differences:

1. In $$\nabla_{\theta} L_{SFT}$$ there is **no reward term** $$r(x, y)$$, which effectively means we can view SFT as having $$r(x, y) = 1$$ for all instruction–response pairs in $$D_{SFT}$$.  
2. The expectation in SFT is taken **only over the dataset samples**, rather than over the full policy distribution as in RL.  

Next, we reformulate the SFT gradient so that its expectation is also taken over the policy distribution, enabling a direct comparison with RL:

$$
\begin{align}
\nabla_{\theta} L_{SFT} &= -\mathbb{E}_{x,y \sim D} [\nabla_{\theta}\log \pi_{\theta}(y|x)] \\
&= -\mathbb{E}_{x\sim D}\mathbb{E}_{y \sim \delta_{y^*(x)}(y)} [\nabla_{\theta}\log \pi_{\theta}(y|x)]
\end{align}
$$

Here, we decompose the expectation over the SFT dataset $$D$$ into an expectation over prompts $$x$$ drawn from $$D$$, and an inner expectation over a **Dirac distribution** $$\delta_{y^*(x)}(y)$$, where all the probability mass is concentrated at the ground-truth response $$y^*(x)$$ for a given prompt $$x$$.  

We can now apply **importance sampling** to express the inner expectation in terms of the policy distribution:

$$
\begin{align}
\nabla_{\theta} L_{SFT} &= -\mathbb{E}_{x\sim D}\mathbb{E}_{y \sim \delta_{y^*(x)}(y)} [\nabla_{\theta}\log \pi_{\theta}(y|x)] \\
&= -\mathbb{E}_{x\sim D}\mathbb{E}_{y \sim \pi_{\theta}(.|x)} \left[\frac{\delta_{y^*(x)}(y)}{\pi_{\theta}(y|x)} \nabla_{\theta}\log \pi_{\theta}(y|x)\right]
\end{align}
$$

From this, we see that an additional **weight term** emerges:

$$
w = \frac{\delta_{y^*(x)}(y)}{\pi_{\theta}(y|x)} = \frac{1}{\pi_{\theta}(y^*(x)|x)}
$$

for each SFT pair $$(x, y^*(x))$$.  
This weight $$w$$ can be interpreted as an **implicit reward** for the instruction–response pair once the SFT gradient is expressed over the policy distribution.  

However, if a given SFT example $$(x, y^*(x))$$ has **low probability** under the current policy $$\pi_{\theta}$$, then $$\frac{1}{\pi_{\theta}(y^*(x)|x)}$$ becomes **large**, leading to disproportionately high values of $$w$$.  
This results in an **ill-posed reward structure** for SFT, where extremely large gradient magnitudes can arise around specific dataset points, ultimately causing **entropy collapse** and greater **forgetting** compared to RL.

In this context, let us see why RL forgets less by revisiting the RL loss gradient. 

$$
\nabla_{\theta} L_{RL} = -\mathbb{E}_{x \sim D}\mathbb{E}_{y \sim \pi_{\theta}(.|x)}[\nabla_{\theta} \log \pi_{\theta}(y|x) \, r(x, y)]
$$


**Reward replaces Dirac weighting**. 
In SFT, the gradient is weighted by a Dirac delta centered at the ground-truth response — effectively forcing the model to assign high probability to a single target output.
In contrast, the RL gradient uses a **reward signal** $$r(x, y)$$ derived from task performance or preference models.
This makes the gradient weight **bounded and meaningful**, rather than inversely proportional to the model’s own probability of producing that response.

**Expectation is over the current policy**. 
RL updates depend on samples drawn from the **current policy** $$\pi_{\theta}(y|x)$$, not on a fixed dataset.
The model thus learns from what it actually generates, making updates **self-consistent** with its behavior.
This avoids the sharp gradient corrections that SFT applies to low-probability dataset responses, leading to smoother learning and less forgetting.

**Entropy regularization**
Most RL objectives, such as PPO and other trust region based methods, include an **entropy term** that promotes exploration and diversity:

$$
L = L_{RL} - \beta \, \mathbb{E}_{x}\mathbb{E}_{y \sim \pi_{\theta}(.|x)}[\log \pi_{\theta}(y|x)]
$$

This **discourages overconfidence** and helps maintain a broader response distribution — further reducing the tendency to collapse or forget previously learned behaviors.

## Modifications to SFT to Consider 

While SFT has its own pitfalls it is necessary to allow the model to follow instructions and setup the stage for RL for human alignment.

The same has been illustrated by the slide from deeplearning.ai, where the pretrained model is not able to follow instructions.

![Picture 1](https://github.com/user-attachments/assets/79b05cc9-e9fa-44f8-a9c8-7d9e69dd37d6)


Figure 1.0 illustrating why SFT is required.

In this context we will discuss two recent approaches to Finetuning.

### Reward Rectification via Dynamic Reweighting

The paper [1] proposes a modified version of **Supervised Fine-Tuning (SFT)** that removes the effect of the unstable implicit reward factor $$w$$ by introducing a corrective scaling term $$\frac{1}{w}$$.  
The key idea is to neutralize the implicit reward magnitude without allowing unwanted gradients to flow through this term.

Since 
$$
\frac{1}{w} = \pi_{\theta}(y|x)
$$

the corrective factor $$\alpha_{\text{correction}}$$ is defined as:

$$
\alpha_{\text{correction}} = sg\!\left(\frac{1}{w}\right) = sg\!\left(\pi_{\theta}(y^*|x)\right),
$$

where the **stop-gradient** operator $$sg$$ prevents any gradients from propagating through the correction factor $$\alpha_{\text{correction}}$$.  
In practice, this corresponds to **`detach()`** in PyTorch or **`tf.stop_gradient()`** in TensorFlow.




Using this correction, the **Dynamic Fine-Tuning (DFT)** loss and its gradient can be written as:

$$
\begin{align}
L_{DFT} &= -\mathbb{E}_{x, y^* \sim D} [\, sg(\pi_{\theta}(y^*|x)) \, \log \pi_{\theta}(y^*|x) \,] \\
\nabla_{\theta} L_{DFT} &= -\mathbb{E}_{x, y^* \sim D} [\, sg(\pi_{\theta}(y^*|x)) \, \nabla_{\theta}\log \pi_{\theta}(y^*|x) \,]
\end{align}
$$

This formulation ensures that the **implicit reward** for all curated responses $$y^*$$ is uniformly **1**, rather than being scaled by $$w$$.  
As a result, the model avoids overemphasizing low-probability dataset responses, leading to **more stable learning**.

This approach is conceptually similar to **Reinforcement Learning with Verifiable Rewards (RLVR)**, where every verified correct response receives the same fixed reward, maintaining stability in updates.

Applying the correction factor to the entire trajectory can cause instability.  
Therefore, in practice, the corrective scaling is applied **at the token level** within each sequence:

$$
L_{DFT} = -\mathbb{E}_{x, y^* \sim D} \Bigg[ \sum_{t=1}^{|y^*|} sg\!\big(\pi_{\theta}(y_{t}^{*} | y_{<t}^{*}, x)\big) \log \pi_{\theta}(y_{t}^{*} | y_{<t}^{*}, x) \Bigg]
$$

This token-wise formulation stabilizes training by ensuring smooth gradient propagation while maintaining consistent reward normalization across tokens.


## Proximal SFT

**Proximal Supervised Fine-Tuning (PSFT)** enhances standard Supervised Fine-Tuning (SFT) by introducing a **trust-region–based optimization strategy**, inspired by **Proximal Policy Optimization (PPO)**.  
This formulation draws a close parallel between SFT and policy-gradient–based reinforcement learning (RL), revealing that SFT can be viewed as a *special case* of RL where the rewards are uniform.



### Connecting SFT and Policy Gradients

In SFT, the model is trained on a fixed dataset of prompt–response pairs $$(x, y^*) \sim D$$.  
The loss minimizes the negative log-likelihood of the correct response:

$$
\begin{align}
\nabla_{\theta} L_{SFT} 
&= -\mathbb{E}_{x, y \sim D} \big[\nabla_{\theta}\log \pi_{\theta}(y|x)\big] \\
&= -\mathbb{E}_{x \sim D} \, \mathbb{E}_{y \sim \delta_{y^*(x)}(y)} \big[\nabla_{\theta}\log \pi_{\theta}(y|x)\big]
\end{align}
$$

Here, the expectation is taken over a **Dirac delta distribution** centered on the gold response $$y^*(x)$$, meaning the model is updated only using the ground-truth sample from the dataset.


### RL Objective Formulation

In contrast, the RL objective optimizes the expected reward under the model’s policy distribution:

$$
\begin{align}
L_{RL1} &= -\mathbb{E}_{x \sim D} \, \mathbb{E}_{y \sim \pi_{\theta}(\cdot|x)} [r(x,y)] \\
\nabla_{\theta} L_{RL} &= -\mathbb{E}_{x \sim D} \, \mathbb{E}_{y \sim \pi_{\theta}(\cdot|x)} 
\big[\nabla_{\theta}\log \pi_{\theta}(y|x) \, r(x,y)\big]
\end{align}
$$

Using the log-derivative trick, the corresponding loss can equivalently be expressed as:

$$
L_{RL2} = -\mathbb{E}_{x \sim D} \, \mathbb{E}_{y \sim \pi_{\theta}(\cdot|x)} 
\big[\log \pi_{\theta}(y|x) \, r(x,y)\big]
$$

Both $$L_{RL1}$$ and $$L_{RL2}$$ share identical gradients, and can thus be considered **conjugate** in the sense that they belong to the same equivalence class with respect to their gradient operator.


### Viewing SFT as a Special Case of RL

From this perspective, **SFT** can be interpreted as a **degenerate case** of the RL objective, where:

- The reward $$r(x,y) = 1$$ for all prompt–response pairs, and  
- The expectation over responses is replaced by a deterministic distribution centered on $$y^*(x)$$.

Hence, the SFT loss structurally matches $$L_{RL2}$$, differing only in the **sampling distribution**:

$$
\text{SFT: } \mathbb{E}_{y \sim \delta_{y^*(x)}}[\cdot]
\quad \text{vs.} \quad
\text{RL: } \mathbb{E}_{y \sim \pi_{\theta}(\cdot|x)}[\cdot]
$$

This equivalence forms the foundation for **Proximal SFT**, which augments the SFT objective with a *PPO-style trust region* to ensure stable updates while preserving the efficiency of supervised training.

















  
## References

[1] [ON THE GENERALIZATION OF SFT: A REINFORCEMENT LEARNING PERSPECTIVE WITH REWARD RECTIFICATION](https://arxiv.org/pdf/2508.05629)  






