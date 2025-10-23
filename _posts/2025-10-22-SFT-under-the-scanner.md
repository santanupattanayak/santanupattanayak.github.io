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


