---
layout: post
title: "Stochastic Gradient Descent Through the Lens of Robbins Monro Algorithm"
date: 2025-07-14 00:00:00 -0000
author: Santanu Pattanayak
tags: banach fixed point, Contraction Mapping, gradient descent convergence through Contraction Mapping. 
---

## Introduction

* **Stochastic Gradient Descent** is a variant of Gradient Descent where instead of computing the Gradient over an entire dataset at all iterations, we compute the gradient over a set of randomly sampled datapoints. When the randomly sampled datapoints in each iteration is $$1$$ it's called Stochastic Gradient Descent whereas when the number of points is some $$m << N$$ where $$N$$ is the number of training datapoints its called a Minibatch Gradient Descent.

$$
\begin{align}
\lvert x_{m} - x_{n} \rvert &= \lvert Tx_{m-1} - Tx_{n-1} \rvert \\
&\le c\lvert x_{m-1} - x_{n-1} \rvert \\
&\le c^{n}\lvert x_{m-n} - x_{0} \rvert \\
L(\theta) = {\mathbb{E_{x,y}}
\end{align}
$$

