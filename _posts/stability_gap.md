---
layout: post
title: "Stability Gap in Continual Learning and Mitigation Strategies"
date: 2025-10-22 00:00:00 -0000
author: Santanu Pattanayak
tags: MachineLearning.PPO,SFT,PSFT,LLMs,DeepLearning,Research 
---

# Table of Contents
1. [Introduction](#introduction)
2. [The Modern Training Pipeline — Roles of Pretraining, SFT, and Alignment](#pipeline)
3. [Mathematical Explanation as to why SFT forgets more than RL](#forgets)
4. [Why SFT Still Matters](#sft)
5. [Reward Rectification via Dynamic Reweighting](#dft)
6. [Proximal SFT](#pfst)
7. [Conclusion](#conclusion)


## Introduction

Artificial neural networks excel when trained on large static datasets—but begin to fail when learning multiple tasks sequentially. This challenge is known as Continual Learning (CL). The most fundamental barrier in CL is the stability–plasticity dilemma:
Plasticity: A model should adapt quickly to new tasks.
Stability: A model should retain previously learned tasks.
In practice, deep networks often become too plastic—overwriting past knowledge as soon as they learn a new task. This leads to catastrophic forgetting.
In recent years, researchers have introduced techniques that explicitly restore stability while maintaining plasticity. However, a subtle but important problem remains: even with these mitigation strategies, models show a stability gap.