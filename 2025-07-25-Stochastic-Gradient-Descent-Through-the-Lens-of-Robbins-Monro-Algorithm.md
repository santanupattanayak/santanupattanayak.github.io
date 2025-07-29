---
layout: post
title: "Stochastic Gradient Descent Through the Lens of Robbins Monro Algorithm"
date: 2025-07-23 00:00:00 -0000
author: Santanu Pattanayak
tags: Gradient Descent, Stochastic Gradient Descent, Robbins Monro Algorithm 
---

## Introduction

* **Stochastic Gradient Descent** is a variant of Gradient Descent where instead of computing the Gradient over an entire dataset at all iterations, we compute the gradient over a set of **randomly sampled datapoints**. When the randomly sampled datapoints in each iteration is $$1$$ it's called Stochastic Gradient Descent whereas when the number of points is some $$m << N$$ where $$N$$ is the number of training datapoints its called a Minibatch Gradient Descent.

* If the loss for the datapoint $$x,y$$ is denoted by $$l(x,y)$$ then the total loss considering entire data distribution is as follows:
  
$$
\begin{align}
L(\theta) &= \mathop {\mathbb E}_{x,y \sim P(x,y)} \[l(\theta,x,y)\] \\
\end{align}
$$

* The gradient of the loss naturally can be expressed as the sum or integral over the gradient of the individual datapoints $$x,y$$. The same can expressed as

$$
\begin{align}
\nabla_{\theta}L(\theta) &= {\mathop {\mathbb E}_{x,y \sim P(x,y)}} \nabla_{\theta}L(\theta)  
\end{align}
$$





