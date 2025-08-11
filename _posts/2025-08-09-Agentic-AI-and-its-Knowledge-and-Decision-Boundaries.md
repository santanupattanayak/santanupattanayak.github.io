---
layout: post
title: "Agentic AI and its Knowledge and Decision Boundary"
date: 2025-08-09 00:00:00 -0000
author: Santanu Pattanayak
tags: Reinforcement Learning, RL, Alignment in LLMs, RLHF  
---

# Table of Contents
1. [Introduction to Agentic AI](#introduction)



## Introduction to Agentic AI<a name="introduction"></a>

Up until now, most AI models have been **passive assistants** — we provide an input or prompt, they generate an output, and the interaction ends.
**Agentic AI** represents a shift from this reactive pattern to a proactive one. In this framework, systems are built to **act autonomously**: reasoning through problems, taking multiple sequential steps, interacting with external tools or environments, and adapting their approach based on feedback.

Much like a capable human assistant, an agentic AI’s goal is not just to respond, but to **actively work toward achieving a specific objective**.

In our discussion, we focus mainly on **LLM-based agents**. Given a task, such an agent attempts to accomplish it through a trajectory of **internal reasoning** and **external tool usage**. At each step, the LLM agent must determine:

1. What additional information is required to solve the task.
2. Whether this information can be obtained through internal reasoning techniques — such as **Chain of Thought** or **Tree of Thought** — or if it requires querying external tools.
As a general principle, the agent should only resort to external tools when the necessary knowledge lies **outside the model’s parametric space**.