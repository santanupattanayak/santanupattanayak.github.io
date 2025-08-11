---
layout: post
title: "Agentic AI and its Knowledge and Decision Boundary"
date: 2025-08-09 00:00:00 -0000
author: Santanu Pattanayak
tags: Reinforcement Learning, RL, Alignment in LLMs, RLHF  
---

# Table of Contents
1. [Introduction to Agentic AI](#introduction)
2. [Proximal Policy Optimization ](#ppo)
3. [Training the Reward Model](#trm)
4. [Reward Model Construction from SFT Model](#Rrmcfsft)
5. [Direct Preference Optimization](#dpo)
6. [Derivation of DPO Objective from PPO and the Reward Model Training loss](#ppo2dpo)
7. [Feature Comparison between PPO style Alignment vs DPO](#ppovsdpo)


## Introduction to Agentic AI<a name="introduction"></a>

Up until now our AI models have been passive assistants — we provide an input or prompt, they produce an output, and the interaction ends there.
Agentic AI marks a shift from this reactive pattern to a more proactive one. In this framework, systems are designed to act autonomously: they can reason through problems, take multiple sequential steps, interact with external tools or environments, and adapt their approach based on feedback.

Much like a capable human assistant, an agentic AI’s objective is not just to respond, but to actively work toward achieving a specified goal.

Mostly the agents we are going to talk about are LLM based - where the LLM agent given a task tries to accomplish the same through a trajectory of internal reasoning and external tool usage. At every step the LLM agent is supposed to reason what additional information is required to solve a task and whether additional information can we achieved through internal reasoning using techniques such as  Chain of Thought and Tree of thought or the agent needs to take help of external tools to acquire the same. In general the LLM agent should only take help of a tool if the information is not available within the parametric space of the model.
