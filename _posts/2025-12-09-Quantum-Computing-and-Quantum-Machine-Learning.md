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

A **qubit**, on the other hand, is a two-state quantum system that can exist in a superposition of both 0 and 1 simultaneously. The basis states 0 and 1 form an orthogonal basis, typically represented as the vectors $$[1, 0]^{T}$$ and $$[0, 1]^{T} $$. In quantum mechanics, vectors live in a complex Hilbert space and are expressed using **ket notation**. Thus, we write these basis states as `|0⟩` and `|1⟩`.

A general qubit state is a linear combination (superposition) of these basis states:

$$
|\phi\rangle = \alpha |0\rangle + \beta |1\rangle
$$

where $$|\alpha|^2$$ is the probability of measuring the system in state $$|0\rangle$$, and $$|\beta|^2$$ is the probability of measuring it in state $$|1\rangle$$.  
It is important to emphasize that these probabilities **do not** imply the qubit is secretly in one of the two states. Prior to measurement, the qubit genuinely exists in a superposition of both. The probabilities only describe the outcomes **when we finally perform a measurement**.

The co-efficients $$\alpha$$ and $$\beta$$ are complex numbers and they are referred to as probability amplitudes. The state $$\ket{\phi}$$ can be written as the vector $$[\alpha. \beta]^{T}$$ where the first dimension corresponds to basis $$\ket{0}$$ while the second dimension corresponds to  $$\ket{1}$$.

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



We can re-write the general superposition qubit state $$\ket{\phi} = \alpha \ket{0} + \beta \ket{1} $$ in a different basis altogether as shown below 

$$
\begin{align}
|\phi\rangle &= \alpha |0\rangle + \beta |1\rangle \\
&= \frac{(\alpha + \beta)}{\sqrt(2)} \frac{(\ket{0} + \ket{1})}{\sqrt(2)} + \frac{(\alpha - \beta)}{\sqrt(2)} \frac{(\ket{0} - \ket{1})}{\sqrt(2)} \\
&= \frac{(\alpha + \beta)}{\sqrt(2)} \ket{+} + \frac{(\alpha - \beta)}{\sqrt(2)} \ket{-}
\end{align}
$$

Now if we measure qubit in the $$\ket{+},\ket{-}$$ basis then we would observe $$\ket{+}$$ with probability $$\frac{|\alpha + \beta|^{2}}{2}$$ and $$\ket{-}$$ with probability $$\frac{|\alpha - \beta|^{2}}{2}$$.
This illustrates the fact the same vector can collapse to a different set of basis vectors based on the basis used for measurement.




## Realization of a Qubit
To build intuition for qubit basis states, consider an electron’s **spin**. The *spin-up* state can be associated with $$\ket{0}$$, while the *spin-down* state corresponds to $$\ket{1}$$. This provides one concrete physical realization of how qubits can be implemented.

Also as per the atomic model an electron can exist in one of the **ground state** or in one of the remaining energy state which we collectively call as the **excited state**. Ground state can be denoted by $$\ket{0}$$ which the excited state can be denoted as $$\ket{1}$$. 

By projecting light on an electron for an appropriate amount of time an electron in ground state $$\ket{0}$$ can be moved to an excited state $$\ket{1}$$ and vice versa. 
An electron can be moved to a superposition state of $$\ket{0}$$ and $$\ket{1}$$ by reducing the duration of the time light is projected on to the atom.


## Quantum Correlation through Entanglement 

Quantum states involving multiple qubits can exhibit strong correlations arising from a uniquely quantum property known as **entanglement**. A general two-qubit state can be written as:

$$
\ket{\phi} = \alpha_{00}\ket{00} + \alpha_{01}\ket{01} + \alpha_{10}\ket{10} + \alpha_{11}\ket{11}
$$

Here, $$\ket{ij}$$ represents the joint basis state in which the first qubit is in state $$\ket{i}$$ and the second qubit is in state $$\ket{j}$$.

Since measurement must yield exactly one of these basis states, the probability amplitudes must satisfy:

$$
|\alpha_{00}|^{2} + |\alpha_{01}|^{2} + |\alpha_{10}|^{2} + |\alpha_{11}|^{2} = 1.
$$

If we choose $$\alpha_{00} = \alpha_{11} = \frac{1}{\sqrt{2}}, \qquad \alpha_{01} = \alpha_{10} = 0,$$  
we obtain the well-known **Bell state**:

$$
\ket{\phi} = \frac{1}{\sqrt{2}}(\ket{00} + \ket{11})
$$

Suppose we measure the first qubit. If the outcome collapses it to $$\ket{0}$$, then the post-measurement joint state becomes:

$$
\ket{\psi} = \ket{00}
$$

Given this outcome, measuring the second qubit will **always** yield $$\ket{0}$$—with probability 1—because in this Bell state the only component consistent with the first qubit being in $$\ket{0}$$ is the joint state $$\ket{00}$$.

This demonstrates how entanglement creates **strong, non-classical correlations** between qubits, where measuring one qubit instantaneously determines the state of the other, regardless of physical separation.

To illustrate a physical example of quantum entanglement, consider a quantum system with **zero total angular momentum** that emits two photons, $$P_1$$ and $$P_2$$. Photons possess spin, and because angular momentum must be conserved, if one photon is in spin-up state (represented by $$\ket{0}$$), the other must be in  spin-down state (represented by $$\ket{1}$$), and vice versa.

Since each photon has an equal tendency to be in either spin state, the joint state of the two photons is described by the entangled state:

$$
\ket{\phi} = \frac{1}{\sqrt{2}}(\ket{01} + \ket{10})
$$

If we measure one photon and find it in the spin-up state, we immediately know **with probability 1** that the other photon is in the spin-down state. This perfect correlation is a hallmark of quantum entanglement.


## Quantum Gates

Much like how classical gates alter the state of the bits Quantum gates are designed to alter the state of the Quantum states. Quantum gates are unitary operations that transform qubit states while preserving their overall probability. Below, we explore some fundamental 1-qubit gates—**X**, **Hadamard**, and **Rotation** gates—and a key 2-qubit gate, the **CNOT** gate. These gates form the building blocks of all quantum circuits.



### 1. X Gate 

The **X gate** acts like a quantum version of the classical NOT gate. It flips the state of a qubit:

$$
X = \begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
$$

#### Effect on Basis States

$$
X\ket{0} = \ket{1}
$$

$$
X\ket{1} = \ket{0}
$$

As we can see the **X gate** swaps the amplitudes of the basis states.



### 2. Hadamard Gate (H Gate)

The **Hadamard gate** maps basis states to coherent combinations of $$\ket{0}$$ and $$\ket{1}$$.

$$
H = \frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 1 \\
1 & -1
\end{bmatrix}
$$

#### Effect on Basis States

$$
H\ket{0} = \frac{1}{\sqrt{2}}(\ket{0} + \ket{1})
$$

$$
H\ket{1} = \frac{1}{\sqrt{2}}(\ket{0} - \ket{1})
$$

The Hadamard gate is crucial for quantum parallelism, creating states that encode multiple possibilities simultaneously.

### 3. Rotation Gates

Rotation gates rotate the state of a qubit and allow for fine-grained, continuous transformations.

### Rotation Around X-axis  
$$
R_x(\theta) = 
\begin{bmatrix}
\cos(\theta/2) & -i\sin(\theta/2) \\
-i\sin(\theta/2) & \cos(\theta/2)
\end{bmatrix}
$$

### Rotation Around Y-axis  
$$
R_y(\theta) = 
\begin{bmatrix}
\cos(\theta/2) & -\sin(\theta/2) \\
\sin(\theta/2) & \cos(\theta/2)
\end{bmatrix}
$$

### Rotation Around Z-axis  
$$
R_z(\theta) =
\begin{bmatrix}
e^{-i\theta/2} & 0 \\
0 & e^{i\theta/2}
\end{bmatrix}
$$

### Effect on Basis States ( Example: $$R_z(\theta)$$ )

$$
R_z(\theta)\ket{0} = e^{-i\theta/2}\ket{0}
$$

$$
R_z(\theta)\ket{1} = e^{i\theta/2}\ket{1}
$$

Rotation gates introduce controlled phase and amplitude adjustments—essential for building arbitrary single-qubit operations. We can create any desired state for a qubit in the 2-dimensional Hilbert space by suitable combination of these 3 rotation gates.


### 4. CNOT Gate (Controlled-NOT)

The **CNOT gate** is a 2-qubit gate that flips the target qubit *only if* the control qubit is in state $$\ket{1}$$. Its matrix form (with first qubit as control) is:

$$
\text{CNOT} =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0
\end{bmatrix}
$$

#### Effect on Basis States

$$
\text{CNOT}\ket{00} = \ket{00}
$$

$$
\text{CNOT}\ket{01} = \ket{01}
$$

$$
\text{CNOT}\ket{10} = \ket{11}
$$

$$
\text{CNOT}\ket{11} = \ket{10}
$$


The **X**, **H**, and **rotation gates** form a universal set for single-qubit transformations. With the **CNOT** gate included, we obtain a universal gate set capable of constructing *any* quantum computation. The ability of CNOT to create entanglement is essential for quantum algorithms, quantum teleportation, and quantum error correction.
