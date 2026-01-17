---
layout: post
title: "Introduction to Quantum Computing and Quantum Machine Learning"
date: 2025-12-09 00:00:00 -0000
author: Santanu Pattanayak
tags: Quantum Computing, Quantum Machine Learning 
---

# Table of Contents
1. [Introduction](#introduction)
2. [Quantum Bit (Qubit)](#qubit)
3. [Measurement](#measurement)
4. [Realization of a Qubit](#real)
5. [Quantum Correlation through Entanglement](#corr)
6. [Quantum Gates](#gates)
7. [Quantum Interference](#interference)
8. [Quantum Algorithms Leverage Superposition, Entanglement, and Interference](#bigpic)
7. [Conclusion](#conclusion)

## Introduction <a name="introduction"></a>
While the intriguing properties of superposition, entanglement, and interference promise significant computational advantages, quantum computing is not a universal replacement for classical computing. Instead, its true potential lies in identifying specific classes of problems where quantum algorithms can offer a genuine Quantum Advantage over classical approaches.
In this blog post, we explore several algorithms within this paradigm—some that are expected to deliver quantum advantage in the long run, and others that are already beginning to demonstrate practical value today. 

## Grover's Algorithm for Database Search <a name="grover"></a>

Let us first look into Grover's search algorithm where given a database of $$N$$ elements, we want to return the index of the target $$x$$ satisfying certain criteria. Let's say if the criteria is defined by a function $$f$$ we would like to find the $$\hat{x}$$ such that 

$$
f(\hat{x}) = 1
$$

Here is how the algorithm proceeds

* The $$N$$ choices can be represented by $$n=\log_2{N}$$ qubits, and we can start with an equal superposition state of all the $$N$$ states. 
This can be achieved by applying Hadamard gate $$H$$ on all the $$n$$ qubits such that 

$$
\ket{\phi_{0}} = H^{\otimes n} \ket{0}^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{x \in \{0, 1\}^n} \ket{x}
$$

Let the target be represented as the winner $$\ket{w}$$ and linear combination of all the other  $$2^{n} - 1 =N-1$$ states be represented as the loser state $$ket{l}$$

$$
\ket{\phi_{0}} = \frac{1}{\sqrt{2^n}} \sum_{x \in \{0, 1\}^n} \ket{x} \\
&=  \frac{\sqrt{2^n -1}}{\sqrt{2^n}} \frac{1}{\sqrt{2^n -1}}\sum_{x \ne w } \ket{x} + \frac{1}{\sqrt{2^n}}\ket{w} \\
&=   \frac{\sqrt{2^n -1}}{\sqrt{2^n}} \ket{l} +  \frac{1}{\sqrt{2^n}}\ket{w}
$$


* Next, a Quantum Circuit applies  the Unitary transform $$U_f$$ to the state $$\ket{\phi_{0}}$$ to implement the function $$f$$ on each of the basis state.

$$
U_f \frac{1}{\sqrt{2^n}} \sum_{x \in \{0, 1\}^n} \ket{x} = \frac{1}{\sqrt{2^n}} \sum_{x \in \{0, 1\}^n} (-1)^{f(x)} \ket{x}
$$

Since for the winning solution 

