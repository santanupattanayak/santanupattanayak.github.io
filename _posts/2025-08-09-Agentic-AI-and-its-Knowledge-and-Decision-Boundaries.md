---
layout: post
title: "Agentic AI and its Knowledge and Decision Boundary"
date: 2025-08-09 00:00:00 -0000
author: Santanu Pattanayak
tags: Reinforcement Learning, RL, Alignment in LLMs, RLHF  
---

# Table of Contents
1. [Introduction to Agentic AI](#introduction)
2. [Knowledge Boundary and Decision Boundary of an Agent](#knowledge-decision-boundary)
3. [Misalignment between Knowledge and Decision Boundary](#knowledge-decision-misalign)



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

## Knowledge Boundary and Decision Boundary of an Agent <a name="knowledge-decision-boundary"></a>

Each AI model $$M$$ has knowledge compressed within its parametric space generally acquired during Model pre-training. Such knowledge can be retrieved using a combination of **internal reasoning methodologies** such as Chain of thought(CoT) [1] , Tree of Thought [2], Reflection [3].
Also, each model $$M$$ has its own understanding of what it knows and what it doesn't and hence a decision boundary of whether it should use internal reasoning or take help from external tools. If the **model is self-conscious**, then the **knowledge boundary should match the decision boundary**. The decision boundary generally is tuned during **Supervised Finetuning(SFT)** and during **Alignment using RLHF**.
The paper **Toward a Theory of Agents as Tool-Use Decision-Makers** [4] defines the Knowledge and Decision boundary very aptly.

Let for a given time step $$t$$, let $$W$$ represent the complete set of world knowledge. The author's define the internal and external knowledge of model $$M$$ as:

$$
\begin{align}
K_{int}(M,t) &\subseteq W \\
K_{ext}(M,t) &=W - K_{int}(M,t)
\end{align}
$$

where $$K_{int}(M,t)$$ is the internal knowledge embedded in the model $$M$$ while $$K_{ext}(M,t)$$ represents the external information available from the world. The knowledge boundary is the separation between the two:   

$$
\begin{align}
\partial K(M,t) = \partial K_{int}(M,t) = \partial K_{ext}(M,t)
\end{align}
$$

Now to define decision boundary we consider the internal reasoning methodologies like Chain of Thought, Tree of Thought, etc as internal reasoning tools as formalized in [4]
If the model at any time $$t$$ can take help of $$m$$ internal reasoning tools $$T_{int} = \{t_{int}^{0},t_{int}^{1},..,t_{int}^{m}\}$$ and $$n$$ external tools $$T_{ext} = \{t_{ext}^{0},t_{ext}^{1},..,t_{ext}^{n}\}$$ then the decision boundary $$\partial D(M,t)$$ is the point at which the model decides whether to use the internal reasoning tools $$T_{int}$$ or the external tools $$T_{ext}$$.  

$$
\begin{align}
\partial D(M,t) = \partial T_{int}(M,t) = \partial T_{ext}(M,t)
\end{align}
$$

As stated earlier the Knowledge boundary and the Decision boundary of the Model $$M$$ should match for optimal performance, where the external tools from $$T_ext$$ would only be called when required.

<img width="770" height="285" alt="image" src="https://github.com/user-attachments/assets/6f0c806c-2c5b-4e84-aee2-aaacb61d4889" />


Figure 1. Knowledge and Decision Boundary of Self Aware Agent.

## Misalignment between Knowledge and Decision Boundary <a name="knowledge-decision-misalign"></a>

In general the AI Models are **not self-aware** as their  knowledge and decision boundaries **don't align**.  This leads to two problems:
- When the model has the information stored in its parametric space but still calls an external tool to get the information, excessive tool calling takes place. In this case the model underestimates its knowledge. 
- When the model overestimates the knowledge that it doesn't have, the output responses suffer from hallucination.

<img width="770" height="285" alt="image" src="https://github.com/user-attachments/assets/7c59a857-fe20-4021-be1e-aa53de4119fd" />

Figure 2. Misaligned Knowledge and Decision Boundary

We can see in Figure 2. the knowledge boundary and the decision boundary doesn't match which leads to suboptimal tool calling and performance. The model $$M$$ has the potential to answer the question $$q_1$$ using internal reasoning methods such as COT, Tree of Thought, etc. as the query is **within the knowledge boundary** of the model. However since it falls outside of the Decision boundary it would take help of the external tools to answer the question when it's not really required.
Similarly, for query $$q_3$$ the **answer is not there in the model parametric space** as it falls **outside the knowledge boundary**, however since the query falls within the decision boundary, the genAI model would not go for tool calling when its actually required. This might lead to hallucinations.


Generally, until now, models have acquired knowledge primarily during the pretraining phase, which did not involve any tool usage. In the supervised fine-tuning (SFT) stage, models learn to better align their decision boundary with their knowledge boundary through curated, tool-calling–specific examples, followed by alignment via RLHF.
However, much of this alignment has focused on correctness rather than tool-use efficiency, often leading to overuse of both internal and external tools.

An optimal agent should balance and optimize for:

- **Internal reasoning methods** such as Chain of Thought (CoT), Tree of Thought (ToT), and Reflection.
- **External tools** for acquiring information beyond its parametric knowledge.

Over-optimizing for internal reasoning can be detrimental, especially for highly complex tasks. Therefore, in training agentic models, a key objective of SFT and alignment is to **reduce unnecessary reliance on external tools while preserving task success**.

## Techniques to align decision boundary to the knowledge boundary <a name="knowledge-decision-align"></a>

The authors of [4] advocates a modification of Pretraining, Supervised Finetuning and Reinforcement Learning for better aligning the Knowledge boundary with the Decision boundary.


### Agentic Pretraining

- Next-token prediction helps model compress world knowledge into a model’s parametric space, but it doesn’t teach models to acquire new knowledge through interaction. 
- Instead, we should move towards **next-tool prediction** — training the model to decide the most appropriate external tool to invoke at each step to acquire appropriate external knowledge. 
- This makes interaction a first-class learning objective, enabling agents to actively seek information they lack. 
- Treating all forms of interaction (API calls, UI navigation, environment manipulation) as structured outputs paves the way for a new scaling law — one that measures knowledge acquisition, not just compression.

### Agentic Supervised Finetuning

- Agents are typically taught to use external tools **during supervised fine-tuning on task-specific datasets**. A key issue with this approach is that using the **same dataset across all models implicitly assumes a uniform knowledge boundary** — which is rarely true.
-  For example, a model $$A$$ with more parameters and trained on a larger pretraining corpus will generally have a broader knowledge boundary than a smaller model $$B$$ trained with fewer parameters and less data. In such cases, applying the same SFT dataset for tool calling is suboptimal. 
- Hence, SFT datasets should be designed with **each model’s knowledge boundary in mind**, ensuring that every model receives its own tailored dataset.

### Agentic Reinforcement Learning 

- Reinforcement Learning (RL) offers a stronger framework for enabling models to understand their knowledge boundary by **allowing them to make mistakes and learn to adaptively align their decision boundary to their knowledge boundary**.
- In this context, a traditional **reward function that values only correctness is insufficient** — the reward must also account for **tool-calling efficiency**.
- For a prompt $$x$$ with multiple correct completions $$\{y_1,y_2,......,y_m\}$$ and corresponding tool calling counts $$\{n_1,n_2,....,n_m\}$$,  the RL objective should guide the LLM to prefer the completion requiring the fewest tool calls, i.e.,
 $$
 \begin{align}
  n_{min} = \min(n_1,n_2,....,n_m)
 \end{align}
 $$


## References

[1] Chain-of-Thought Prompting Elicits Reasoning in Large Language Models : https://arxiv.org/abs/2201.11903  

[2] Tree of Thoughts: Deliberate Problem Solving with Large Language Models : https://arxiv.org/pdf/2305.10601  

[3] Self-Reflection in LLM Agents: Effects on Problem-Solving Performance : https://arxiv.org/abs/2405.06682  

[4] Toward a Theory of Agents as Tool-Use Decision-Makers : https://arxiv.org/pdf/2506.00886v1
