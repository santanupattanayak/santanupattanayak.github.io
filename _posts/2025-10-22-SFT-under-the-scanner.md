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


## From Pretraining to Alignment: Understanding the Roles of Each Stage


## The Modern Training Pipeline — Roles of Pretraining, SFT, and Alignment

| **Stage** | **Primary Objective / Loss Function** | **Data Source** | **Goal / Outcome** | **Key Challenges** |
|------------|--------------------------------------|------------------|--------------------|--------------------|
| **Pretraining** | Minimize language modeling loss:<br>$$\mathcal{L}_{\text{pre}} = -\mathbb{E}_{(x,y)\sim D_{\text{corpus}}} [\log P_\theta(y|x)]$$ | Massive unlabeled text corpora (web, books, code, etc.) | Learn broad linguistic, factual, and structural knowledge; capture general world representations. | High compute cost; biases in data; lack of grounding to human intent or task-level understanding. |
| **Supervised Fine-Tuning (SFT)** | Minimize supervised imitation loss:<br>$$\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(x,y^*)\sim D_{\text{inst}}} [\log P_\theta(y^*|x)]$$ | Curated instruction–response datasets (human-written or synthetic) | Teach the model to follow instructions, structure responses, and stay aligned with factual or stylistic norms. | Distribution collapse around demonstrated behaviors; loss of diversity; reduced generalization to unseen prompts. |
| **Alignment (RLHF / PPO)** | Maximize preference-weighted reward:<br>$$\mathcal{L}_{\text{RL}} = -\mathbb{E}_{x\sim D,\ y\sim \pi_\theta} [r(y|x)] + \beta\, D_{KL}(\pi_\theta \Vert \pi_{\text{ref}})$$ | Human or AI preference signals via pairwise comparisons or reward models | Align responses with human values, tone, and intent while preserving pretrained diversity and fluency. | Reward hacking; instability in optimization; balancing reward improvement with knowledge retention. |

---

Each stage in the pipeline serves a distinct purpose — **pretraining** builds the foundation of knowledge, **SFT** shapes that knowledge into coherent task-following behavior, and **RL-based alignment** fine-tunes those behaviors to better reflect human preferences. Together, they form a sequential refinement process that converts a general-purpose model into a safe and useful assistant.


