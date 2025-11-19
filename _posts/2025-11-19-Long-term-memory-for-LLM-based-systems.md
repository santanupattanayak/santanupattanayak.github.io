---
layout: post
title: "Long term memory for LLM based systems"
date: 2025-10-22 00:00:00 -0000
author: Santanu Pattanayak
tags: MachineLearning.LLM memory, DeepLearning,Research 
---


## Introduction
Large Language Models (LLMs) have revolutionized conversational AI, but they falter when it comes to **long-term coherence**. Most agents forget what you said last week — or even five minutes ago. Enter **Mem0**, a memory-centric architecture that transforms ephemeral chats into persistent, structured knowledge.

This post explores Mem0’s **graph-based memory mechanism**, breaking down its mathematical foundations and practical implications for building production-ready AI agents.

---

## Why Memory Matters in AI Agents
LLMs operate within a **fixed context window**, making them great at short-term reasoning but poor at **multi-session continuity**. Without memory, agents:
- Repeat questions
- Forget user preferences
- Lose track of goals

Mem0 addresses this by introducing a **scalable, structured memory layer** that persists across sessions.

---

## Graph-Based Memory: The Core Idea

Mem0 models memory as a **directed labeled graph**:



\[
\mathcal{G} = (\mathcal{V}, \mathcal{E})
\]



- **Nodes (\(\mathcal{V}\))** represent entities or facts (e.g., “Santanu likes Itô calculus”).
- **Edges (\(\mathcal{E}\))** represent semantic or temporal relations (e.g., “prefers”, “authored”).

This structure enables relational reasoning, efficient retrieval, and scalable updates.

---

## Mathematical Breakdown

### 1. Node Construction
From a conversation transcript \(C = \{u_1, u_2, \dots, u_T\}\), Mem0 extracts salient facts:



\[
f_{\text{extract}}: C \to \mathcal{V}
\]



Each node is embedded as:



\[
\mathbf{v}_i \in \mathbb{R}^d
\]



where \(d\) is the embedding dimension.

---

### 2. Edge Formation
Relations between nodes are predicted via:



\[
r = f_{\text{rel}}(\mathbf{v}_i, \mathbf{v}_j)
\]



yielding edges:



\[
e_{ij} = (\mathbf{v}_i, r, \mathbf{v}_j)
\]



---

### 3. Retrieval via Graph Traversal
Given a query \(q\), Mem0:
1. Encodes it: \(\mathbf{q}\)
2. Finds top-\(k\) similar nodes via cosine similarity
3. Expands to neighbors:



\[
\mathcal{N}(v_i) = \{ v_j \mid (v_i, r, v_j) \in \mathcal{E} \}
\]



This ensures context-aware retrieval beyond surface similarity.

---

##  Clarifying a Common Misconception: Embedding Averaging ≠ Reasoning

Once relevant nodes are retrieved, **you don’t need their embeddings anymore**. What matters is their **textual content**, which should be:
- Injected into the LLM prompt as **natural language context**
- Preserved in full fidelity — not compressed or averaged

### Why Averaging Can Be Harmful Post-Retrieval
- **Loss of granularity**: Averaging embeddings blurs distinctions between facts.
- **LLMs are text-first**: They reason over structured or natural language, not latent vectors.

### Correct Flow in Mem0-style Systems

