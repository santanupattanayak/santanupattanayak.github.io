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

* For minibatch Stochastic gradient descent we estimate the gradient from $$m$$ i.i.d datapoints from $$\mathbb{P}(x,y)$$ which we will denote as $$B=\{(x_i,y_i)\}_{1:m}$$  . Taking the expectation of the minibatch loss $$L_m(\theta)$$ gradient over the batch distribution $$\mathbb{P}^{m}(x,y)$$ is equivalent to the expectation over $$\mathbb{P}(x,y)$$ for each of the i.i.d samples in the batch

$$
\begin{align}
&\mathbb{E}_{x,y \sim P(x,y)} \left[\frac{1}{m} \sum_{1:m}\left[ \nabla_{\theta} \, \ell(\theta, x, y) \right]\right] \\
&= \frac{1}{m} \sum_{1:m}\left[\mathbb{E}_{x,y \sim P(x,y)} \left[ \nabla_{\theta} \, \ell(\theta, x, y) \right]\right] \\
&= \frac{1}{m}.m.\nabla_{\theta} L(\theta) = \nabla_{\theta} L(\theta) 
\end{align}
$$

* We will now prove that the minibatch gradient is an unbiased estimator of the finite training dataset gradient. For that we assume that we have $$N$$ iid datapoints from the data distribution $$\mathbb{P}(x,y)$$. The full dataset gradient is 

$$
\begin{align}
\nabla_{\theta} L(\theta)  = \frac{1}{N} \sum_{1:N}\left[ \nabla_{\theta} \, \ell(\theta, x, y) \right] \\
\end{align}
$$

The expectation over the minibatch $$B$$ gradient is equivalent to the expectation over each datapoint gradient in the minibatch as shown below

$$
\begin{align}
&\mathbb{E}_{B} \left[\frac{1}{m} \sum_{1:m}\left[ \nabla_{\theta} \, \ell(\theta, x, y) \right]\right] \\
&= \left[\frac{1}{m} \sum_{1:m} \mathbb{E}\left[ \nabla_{\theta} \, \ell(\theta, x, y) \right]\right] \\
\end{align}
$$

* For the finite dataset of size $$N$$ this expectation over each datapoint gradient is nothing but the full dataset gradient and hence we can simplify the above as 

$$
\begin{align}
&\mathbb{E}_{B} \left[\frac{1}{m} \sum_{1:m}\left[ \nabla_{\theta} \, \ell(\theta, x, y) \right]\right] \\
&= \left[\frac{1}{m} \sum_{1:m} \mathbb{E}\left[ \nabla_{\theta} \, \ell(\theta, x, y) \right]\right] \\
&= \frac{1}{m} \sum_{1:m} \nabla_{\theta} L(\theta)   \\
&= \frac{1}{m} \sum_{1:m}\frac{1}{N} \sum_{1:N}\left[ \nabla_{\theta} \, \ell(\theta, x, y) \right] \\
&= \frac{1}{m}.m \frac{1}{N} \sum_{1:N}\left[ \nabla_{\theta} \, \ell(\theta, x, y) \right] \\
&= \frac{1}{N} \sum_{1:N}\left[ \nabla_{\theta} \, \ell(\theta, x, y) \right] = \nabla_{\theta} L(\theta)
\end{align}
$$

* 
 



 






