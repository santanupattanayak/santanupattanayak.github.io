---
layout: post
title: "SFT under the scanner"
date: 2025-08-09 00:00:00 -0000
author: Santanu Pattanayak
tags: UAT, Universal approximation theorem, Neural networks, 
---

# Table of Contents


## Introduction <a name="introduction"></a>

Supervised Fine-Tuning (SFT) has long been the first line of post-training for large language models. After pretraining on massive unlabeled corpora, models are refined using curated instruction–response pairs, teaching them to follow human-like instructions and stay grounded in factual, stylistic, or ethical norms. This approach is simple, stable, and data-efficient — but it has begun to reveal some serious limitations.
SFT optimizes a cross-entropy loss that forces the model to imitate the provided responses as closely as possible. While this can quickly improve surface-level helpfulness and coherence, it tends to collapse the model’s output distribution around the demonstrated behaviors compromising performance of already learnt good behaviors in pretraining on various other modalities. The result is a model that performs well on seen data but loses diversity, creativity, and sometimes even factual robustness in unseen contexts. This brittleness becomes more visible as models are deployed in open-ended, interactive settings.
In contrast, Reinforcement Learning (RL)–based post-training, especially with methods like Proximal Policy Optimization (PPO), treats the model as a policy that must balance learning from human feedback with preserving prior knowledge. Rather than imitating demonstrations verbatim, PPO aligns the model using reward signals derived from human or AI preferences — allowing it to improve along directions that enhance user satisfaction while still respecting its pretrained distribution.
This shift from imitation (SFT) to alignment (RL) marks a key evolution in how we fine-tune large models. As we’ll explore next, RL methods can better retain generalization, avoid distribution collapse, and align models more stably — even when explicit supervision is sparse or conflicting.

