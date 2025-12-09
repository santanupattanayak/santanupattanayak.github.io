---
layout: post
title: "Quantum Computing and Quantum Machine Learning"
date: 2025-11-18 00:00:00 -0000
author: Santanu Pattanayak
tags: MachineLearning.LLM memory, DeepLearning,Research 
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

Quantum computing and quantum machine learning are often misunderstood or oversimplified in popular articles that attempt to explain them at a surface level. Through this series of blog posts, I aim to demystify the core concepts behind Quantum Computing and Quantum Machine Learning, highlight where they can be genuinely useful in the long run, and showcase applications that are already beginning to benefit from them.  
To appreciate these topics, we must first build a solid understanding of the foundations of quantum computation.

# Quantum Bit aka Qubit 
Let’s begin with the familiar: a **classical bit**. A bit can take one of two possible values—0 or 1—and at any given time it holds exactly one of these values.

A **qubit**, on the other hand, is a two-state quantum system that can exist in a superposition of both 0 and 1 simultaneously. The basis states 0 and 1 form an orthogonal basis, typically represented as the vectors [1, 0] and [0, 1]. In quantum mechanics, vectors live in a complex Hilbert space and are expressed using **ket notation**. Thus, we write these basis states as `|0⟩` and `|1⟩`.

A general qubit state is a linear combination (superposition) of these basis states:

$$
|\phi\rangle = \alpha |0\rangle + \beta |1\rangle
$$

where $|\alpha|^2$ is the probability of measuring the system in state `|0⟩`, and $|\beta|^2$ is the probability of measuring it in state `|1⟩`.  
It is important to emphasize that these probabilities **do not** imply the qubit is secretly in one of the two states. Prior to measurement, the qubit genuinely exists in a superposition of both. The probabilities only describe the outcomes **when we finally perform a measurement**, a concept we will revisit soon.

To get an intuition for qubit basis states, consider an electron’s **spin**. The *spin-up* state can be associated with `|0⟩`, while the *spin-down* state corresponds to `|1⟩`. This physical analogy offers one concrete realization of how qubits are built in practice.



