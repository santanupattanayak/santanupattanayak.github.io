---
layout: post
title: "Alignment in LLMs"
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
\mathbb{E}_{x \sim D_x}\mathbb{E}_{y \sim \pi_{\theta}(y|x)}\left[r_{\phi}(x,y) - \beta.KL(\pi_{\theta}(y|x) || \pi_{\theta_{SFT}}(y|x))\right] 
\end{align}
$$




