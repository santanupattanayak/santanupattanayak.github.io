---
layout: post
title: "SFT under the scanner"
date: 2025-08-09 00:00:00 -0000
author: Santanu Pattanayak
tags: UAT, Universal approximation theorem, Neural networks, 
---

# Table of Contents


## Introduction <a name="introduction"></a>

Supervised Fine-Tuning (SFT) has long been the first stage of post-training for large language models. After pretraining on massive unlabeled corpora, models are refined using curated instruction–response pairs, teaching them to follow human-like instructions and remain grounded in factual, stylistic, or ethical norms. This process is simple, stable, and data-efficient — yet it has begun to reveal some fundamental limitations.  

SFT optimizes a cross-entropy loss that compels the model to imitate the provided responses as closely as possible. While this quickly improves surface-level helpfulness and fluency, it often causes the model’s output distribution to collapse around the demonstrated behaviors — compromising the performance of already well-learned behaviors acquired during pretraining across diverse modalities. The result is a model that performs well on seen data but loses diversity, creativity, and sometimes even factual robustness when operating in open-ended or interactive environments.  

In contrast, Reinforcement Learning (RL)–based post-training, particularly through methods such as Proximal Policy Optimization (PPO), treats the model as a policy that must balance learning from human feedback with preserving its prior knowledge. Rather than imitating demonstrations verbatim, PPO aligns the model using reward signals derived from human or AI preferences — enabling it to improve along directions that enhance user satisfaction while maintaining consistency with its pretrained distribution.  

This shift from **imitation-driven SFT** to **alignment-oriented RL** marks a significant evolution in the fine-tuning paradigm for large models. RL methods have demonstrated stronger capabilities in **preserving generalization**, **avoiding distribution collapse**, and **achieving more stable alignment**, even when supervision is limited or noisy.  

Nevertheless, **SFT continues to play an essential role** as a preconditioning step before RL-based alignment. 



## The Modern Training Pipeline — Roles of Pretraining, SFT, and Alignment


| **Attribute** | **Pretraining** | **Supervised Fine-Tuning (SFT)** | **Alignment (RLHF / PPO)** |
|----------------|-----------------|----------------------------------|-----------------------------|
| **Primary Objective** | Predict the next token in a sequence — learn general patterns of language, reasoning, and world knowledge by minimizing the difference between predicted and actual text. | Learn to imitate high-quality human or curated responses by minimizing cross-entropy loss over instruction–response pairs. | Optimize model behavior using reward signals from human or AI feedback, improving preference alignment while constraining divergence from the reference policy. |
| **Data Source** | Massive unlabeled text corpora (web, books, academic papers, code, etc.). | Curated instruction–response datasets (human-written or synthetically generated). | Preference data — human or AI rankings, comparisons, or reward model outputs. |
| **Goal / Outcome** | Acquire broad linguistic, factual, and structural understanding of the world. | Teach the model to follow instructions, stay factual, and maintain a coherent conversational style. | Refine the model’s responses to align with human intent, tone, and ethical expectations while preserving its pretrained diversity. |
| **Optimization Method** | Gradient descent on the language modeling objective (e.g., AdamW). | Supervised gradient descent minimizing imitation loss. | Policy-gradient optimization (e.g., PPO or related methods) with reward shaping and KL regularization. |
| **Key Challenges** | Expensive training; data bias; lack of grounding to human values or context. | Distribution collapse around demonstrated examples; loss of diversity and generalization. | Reward hacking; optimization instability; tuning balance between reward gain and knowledge retention. |


Each stage contributes uniquely to the model’s final behavior. **Pretraining** builds a foundation of general knowledge, **SFT** organizes this knowledge into structured instruction-following behavior, and **RL-based alignment** fine-tunes those behaviors to better reflect human preferences without eroding the model’s prior understanding.


