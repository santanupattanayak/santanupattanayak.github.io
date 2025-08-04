---
layout: post
title: "Alignment and Reinforcement Learning with Human Feedback in LLMs"
date: 2025-08-02 00:00:00 -0000
author: Santanu Pattanayak
tags: Reinforcement Learning, RL, Alignment in LLMs, RLHF  
---

## Introduction

Alignment represents a shift from traditional likelihood-based training. Rather than simply maximizing the probability of the next token, alignment focuses on steering a language model’s outputs toward human values, preferences, and intended goals. This approach is essential for mitigating issues such as harmful content, logical inconsistencies, and hallucinations—challenges that next-token prediction alone does not address or prevent. Alignment is mostly done with Reinforcement learning under the tag of Reinforcement Learning with Human Feedback (RLHF). 
The most commonly form of RLHF methods used are:  
1. Proximal Policy Optimization(PPO)
2. Direct Preference Optimization(DPO)
3. Group Relative Policy Optimization(GRPO)

## Proximal Policy Optimization 

Proximal Policy Optimization is a trust method Policy gradient method of Reinforcement Learning Technique where the updates to the policy in each step is done in such a way that the policy parameters doesn't change too much from that in the earlier iteration. Given a stochastic policy $$\pi$$ parameterized by $$\theta$$ that maps any state $$s$$ to an action $$a$$ probabilistically the update rule of PPO is given by  
In the context of the LLM alignment we assume that given a query $$x$$ the LLM which acts as a policy $$\pi_{\theta}$$ generates the output $$y$$ stochastically. We assume that we don't want the PPO to shift the RL aligned model parameters $$\theta$$ to be too far from the Supervised fine-tuned(SFT) model parameters  $$\theta_{SFT}$$. If we sample the queries $$x$$ from some dataset $$D_x$$ then as per PPO the optimization objective is as below 

$$
\begin{align}
L(\theta) &= \mathbb{E}_{x \sim D_x}\mathbb{E}_{y \sim \pi_{\theta}(y|x)}\left[r_{\phi}(x,y)\right] - \beta.KL(\pi_{\theta}(y|x) || \pi_{\theta_{SFT}}(y|x)) \\
&= \mathbb{E}_{x \sim D_x}\mathbb{E}_{y \sim \pi_{\theta}(y|x)}\left[r_{\phi}(x,y) - \beta \log\frac{\pi_{\theta}(y|x)}  {\pi_{\theta_{SFT}}(y|x)}\right] 
\end{align}
$$

The hyperparameter $$\beta$$ balances the reward maximization objective and the objective to prevent too much deviation of the model parameters from the SFT model capture through the KL divergence of the policies.
The reward for completion $$y$$ given query $$x$$ which we have denoted by  $$r_{\phi}(x,y)$$ is generally computed from a trained reward model. Since the reward model assigns a score at the end of the completion of $$y$$ there isn't any reward after each token generation.

## Training the Reward Model

Instead of taking the feedback of the users to a completion based on a given query as the reward in alignment or RLHF, a reward model is trained. Here are the steps towards alignment as illustrated in the InstructGPT paper https://arxiv.org/pdf/2203.02155  
1. First step is to sample prompts from a Prompts Dataset and a labeler comes up with the desired output behavior. This data is used for Supervised Finetuning.


2. Second step is where we build a reward model. For a given prompt, several model outputs are sampled. The labelers ranks the outputs from best to worst. This comparison data is used to train the reward model. Generally the SFT model from first step is taken as a starting point for the Reward Model.

3. Finally the SFT model from first step is optimized using the reward model using Reinforcement Learning.


<img width="760" height="493" alt="image" src="https://github.com/user-attachments/assets/db8438df-10a5-4b87-a3f7-6f3e5b11af73" />


Figure 1: Illustration of the SFT training, Reward Model training and Reinforcement learning for Alignment using Reward Model. 

## Reward Model Construction from SFT Model

1. The reward model architecture in used in RLHF pipelines such as InstructGPT or ChatGPT builds on top of the SFT Model architecture and its weights as the starting weights for the Reward Model with one exception - the final layer which gives the next token probability scores over the vocabulary is replaced by a layer that gives the reward the final output. 
2. The input to the reward model is the prompt $$x$$ along with the completion $$y$$ while the output is the reward $$r_{\phi}(x,y)$$
3. Training the Reward Model is on the preference data set $$D$$. Let's say for the prompt $$x$$ the completion $$y^{+}$$ is preferred over the completion $$y^{-}$$  
The reward model is trained with the softmax loss over the two completions $$y^{+}$$ and $$y^{-}$$ as follows:  

$$
\begin{align}
L(\phi) = -\mathbb{E}_{(x,y^{+},y^{-}) \sim D}  \log\left[\frac {\exp(r_{\phi}(x,y^{+})} {\exp(r_{\phi}(x,y^{+}) + \exp(r_{\phi}(x,y^{-})}\right] 
\end{align}
$$

One important aspect to note here, that we are not regressing on the reward $$r(x,y)$$ directly, but rather they act as logits for the completions $$y$$ given the prompt $$x$$. 

## Direct Preference Optimization
Direct Preference Optimization(DPO) is a RL technique for Alignment which skips training a reward model and given preference pairs dataset $$ x,y^{+}, y_D 










