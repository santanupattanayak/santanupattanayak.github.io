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

- What additional information is required to solve the task. 
- Whether this information can be obtained through internal reasoning techniques — such as **Chain of Thought** or **Tree of Thought** — or if it requires querying external tools.
As a general principle, the agent should only resort to external tools when the necessary knowledge lies **outside the model’s parametric space**.

Given a task $$q$$ to an agentic model $$M$$ equipped with access to a set of external tools $$T = \{t_{0},t_{1},......t_{n}\}$$ the reasoning at step $$k$$ of the agent can be represented as a **tool-integrated trajectory** $$\tau_{k}$$ as follows:

$$
\begin{align}
\tau_{k} = (r_0,tc_{0},o_{0}),(r_1,tc_{1},o_{1}),.....,(r_k,tc_{k},o_{k})
\end{align}
$$

Here, each tuple $$(r_i,tc_{i},o_{i})$$ represents:  
- $$r_i$$ : The reasoning process at step $$i$$.  
- $$tc_{i}$$ : The tool called at step $$i$$ (if any).  
- $$o_i$$ : The output returned by the tool at step $$i$$.

If, for some step $$j$$, no tool invocation is required, the variables $$tc_{j}$$ and $$o_{j}$$ can be considered **empty**. The reasoning information from step $$j$$ can be:  
- Integrated into the reasoning of the subsequent step $$j+1$$, or  
- Used directly to produce the **final answer** if it is the last step.

## Knowledge Boundary and Decision Boundary of an Agent 

Each AI model $$M$$ has knowledge compressed within its parametric space generally acquired during Model pre-training. Such knowledge can be retrieved using a combination of **internal reasoning methodologies** such as Chain of thought(CoT) [1] , Tree of Thought [2], Reflection [3].
Also, each model $$M$$ has its own understanding of what it knows and what it doesn't and hence a decision boundary of whether it should use internal reasoning or take help from external tools. If the model is self-conscious, then the knowledge boundary should match the decision boundary. The decision boundary generally is tuned during *Supervised Finetuning(SFT)* and during *Alignment using RLHF*.
The paper *Toward a Theory of Agents as Tool-Use Decision-Makers* [4] defines the Knowledge and Decision boundary very aptly.

Let for a given time step $$t$$, let $$W$$ represent the complete set of world knowledge. The author's define the internal and external knowledge of model $$M$$ as:

$$
\align{begin}
K_{int}(M,t) &\subseteq W \\
K_{ext}(M,t) &W \complement K_{int}(M,t)
\align{end}
$$
where $$K_{int}(M,t)$$ is the internal knowledge embedded in the model $$M$$ while $$K_{ext}(M,t)$$ represents the external information available from the world. The knowledge boundary is the separation between the two:   

$$
\align{begin}
\del K(M,t) = \del K_{int}(M,t) = \del K_{ext}(M,t)
\align{end}
$$



##References

[1] *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* : https://arxiv.org/abs/2201.11903
[2] *Tree of Thoughts: Deliberate Problem Solving with Large Language Models* : https://arxiv.org/pdf/2305.10601
[3] *Self-Reflection in LLM Agents: Effects on Problem-Solving Performance* : https://arxiv.org/abs/2405.06682
[4] *Toward a Theory of Agents as Tool-Use Decision-Makers* : https://arxiv.org/pdf/2506.00886v1
