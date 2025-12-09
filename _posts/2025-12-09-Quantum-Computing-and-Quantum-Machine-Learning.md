---
layout: post
title: "Quantum Computing and Quantum Machine Learning"
date: 2025-12-09 00:00:00 -0000
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

## Quantum Bit (Qubit)

Let’s begin with the familiar: a **classical bit**. A bit can take one of two possible values—0 or 1—and at any given time it holds exactly one of these values.

A **qubit**, on the other hand, is a two-state quantum system that can exist in a superposition of both 0 and 1 simultaneously. The basis states 0 and 1 form an orthogonal basis, typically represented as the vectors `[1, 0]` and `[0, 1]`. In quantum mechanics, vectors live in a complex Hilbert space and are expressed using **ket notation**. Thus, we write these basis states as `|0⟩` and `|1⟩`.

A general qubit state is a linear combination (superposition) of these basis states:

$$
|\phi\rangle = \alpha |0\rangle + \beta |1\rangle
$$

where $$|\alpha|^2$$ is the probability of measuring the system in state $$|0\rangle$$, and $$|\beta|^2$$ is the probability of measuring it in state $$|1\rangle$$.  
It is important to emphasize that these probabilities **do not** imply the qubit is secretly in one of the two states. Prior to measurement, the qubit genuinely exists in a superposition of both. The probabilities only describe the outcomes **when we finally perform a measurement**.

The co-efficients $$\alpha$$ and $$\beta$$ are complex numbers and they are referred to as probability amplitudes. The state $$\ket{\phi}$$ can be written as the vector $$[\alpha \beta]^{T}$$ where the first dimension corresponds to basis $$\ket{0}$$ while the second dimension corresponds to  $$\ket{1}$$.

## Measurement

A qubit—like any other quantum system—does not reveal its superposition state when measured. Instead, measurement causes the state to **collapse** to one of the basis states. Thus, a superposition state $$\ket{\phi}$$ will collapse to either $$\ket{0}$$ or $$\ket{1}$$ during measurement.

For example, consider the qubit state:

$$
|\phi\rangle = \frac{1}{\sqrt{2}} |0\rangle + \frac{1}{\sqrt{2}} |1\rangle
$$

If we prepare 1000 identical copies of this state and measure each one, we would expect to obtain roughly 500 outcomes of $$\ket{0}$$ and 500 outcomes of $$\ket{1}$$. This is because each basis state has probability

$$
\left(\frac{1}{\sqrt{2}}\right)^2 = \frac{1}{2}.
$$

Do note that 
$$
|\phi\rangle = \frac{1}{\sqrt{2}} |0\rangle + \frac{1}{\sqrt{2}} |1\rangle
$$

We can re-write the superposition qubit state in a different basis altogether as shown below 

$$
\begin{align}
|\phi\rangle &= \alpha |0\rangle + \beta |1\rangle \\
&= \frac{(\alpha + \beta)}{\sqrt(2)} \frac{(\ket{0} + \ket{1})}{\sqrt(2)} + \frac{(\alpha - \beta)}{\sqrt(2)} \frac{(\ket{0} - \ket{1})}{\sqrt(2)} \\
&= \frac{(\alpha + \beta)}{\sqrt(2)} \ket{+} + \frac{(\alpha - \beta)}{\sqrt(2)} \ket{-}
\end{align}
$$

Now if we measure qubit in the $$\ket{+},\ket{-}$$ basis then we would observe $$\ket{+}$$ with probability $$\frac{(\alpha + \beta)^{2}}{2}$$ and $$\ket{-}$$ with probability $$\frac{(\alpha - \beta)^{2}}{2}$$.
This illustrates the fact the same vector can collapse to a different set of basis vectors based on the basis used for measurement.




## Realization of a Qubit
To build intuition for qubit basis states, consider an electron’s **spin**. The *spin-up* state can be associated with $$\ket{0}$$, while the *spin-down* state corresponds to $$\ket{1}$$. This provides one concrete physical realization of how qubits can be implemented.

Also as per the atomic model an electron can exist in one of the **ground state** or in one of the remaining energy state which we collectively call as the **excited state**. Ground state can be denoted by $$\ket{0}$$ which the excited state can be denoted as $$\ket{1}$$. 

By projecting light on an electron for an appropriate amount of time an electron in ground state $$\ket{0}$$ can be moved to an excited state $$\ket{1}$$ and vice versa. 
An electron can be moved to a superposition state of $$\ket{0}$$ and $$\ket{1}$$ by reducing the duration of the time light is projected on to the atom.


