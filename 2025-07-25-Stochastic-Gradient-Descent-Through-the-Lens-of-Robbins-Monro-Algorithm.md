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
L(\theta) &= \mathop {\mathbb E}_{x,y \sim P(x,y)} \left[\ell(\theta,x,y)\right] \\
\end{align}
$$

* The gradient of the loss naturally can be expressed as the expectation of the gradient of the individual datapoints $$x,y$$ over the data distribution as shown below

$$
\begin{align}
\nabla_{\theta} L(\theta) = \mathbb{E}_{x,y \sim P(x,y)} \left[ \nabla_{\theta} \, \ell(\theta, x, y) \right]
\end{align}
$$

* In Stochastic gradient descent the entire gradient is approximated by the gradient of the loss of a single datapoint. It is obvious it's an unbiased estimator of the actual dataset gradient as the expectation over the single datapoint loss gradient is equal to the gradient over the entire dataset loss gradient.
 If the gradient of the single datapoint loss is $$\left[ \nabla_{\theta} \, \ell(\theta, x, y) \right]$$ then taking expectation of it over the entire dataset distribution we get

$$
\begin{align}
\mathbb{E}_{x,y \sim P(x,y)} \left[ \nabla_{\theta} \, \ell(\theta, x, y) \right] = \nabla_{\theta} L(\theta) \\
\end{align}
$$

* Similarly for minibatch Stochastic gradient descent we have the estimated gradient from $$m$$ datapoints (see below). And taking the expectation of the minibatch gradient over the entire data distribution again gives is the gradient over the entire data distribution as shown below

$$
\begin{align}
&\mathbb{E}_{x,y \sim P(x,y)} \left[\frac{1}{m} \sum_{1:m}\left[ \nabla_{\theta} \, \ell(\theta, x, y) \right]\right] \\
&= \frac{1}{m} \sum_{1:m}\left[\mathbb{E}_{x,y \sim P(x,y)} \left[ \nabla_{\theta} \, \ell(\theta, x, y) \right]\right] \\
&= \frac{1}{m}.m.\nabla_{\theta} L(\theta) = \nabla_{\theta} L(\theta) 
\end{align}
$$

 






