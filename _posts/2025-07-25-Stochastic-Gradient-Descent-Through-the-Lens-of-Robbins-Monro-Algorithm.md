---
layout: post
title: "Stochastic Gradient Descent Through the Lens of Robbins Monro Algorithm"
date: 2025-07-23 00:00:00 -0000
author: Santanu Pattanayak
tags: Gradient Descent, Stochastic Gradient Descent, Robbins Monro Algorithm 
---

## Introduction to Stochastic Gradient Descent 

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
&\mathbb{E}_{x,y \sim P(x,y)} \left[\frac{1}{m} \sum_{i=1:m}\left[ \nabla_{\theta} \, \ell(\theta, x_i, y_i) \right]\right] \\
&= \frac{1}{m} \sum_{i=1:m}\left[\mathbb{E}_{x_i,y_i \sim \mathbb{P}(x,y)} \left[ \nabla_{\theta} \, \ell(\theta, x_i, y_i) \right]\right] \\
&= \frac{1}{m}.m.\nabla_{\theta} L(\theta) = \nabla_{\theta} L(\theta) 
\end{align}
$$

* We will now prove that the minibatch gradient is an unbiased estimator of the finite training dataset gradient. For that we assume that we have $$N$$ iid datapoints from the data distribution $$\mathbb{P}(x,y)$$. The full dataset gradient is 

$$
\begin{align}
\nabla_{\theta} L(\theta)  = \frac{1}{N} \sum_{i=1:N}\left[ \nabla_{\theta} \, \ell(\theta, x_i, y_i) \right] \\
\end{align}
$$

The expectation over the minibatch $$B$$ gradient is equivalent to the expectation over each datapoint gradient in the minibatch as shown below

$$
\begin{align}
&\mathbb{E}_{B} \left[\frac{1}{m} \sum_{i=1:m}\left[ \nabla_{\theta} \, \ell(\theta, x_i, y_i) \right]\right] \\
&= \left[\frac{1}{m} \sum_{i=1:m} \mathbb{E}\left[ \nabla_{\theta} \, \ell(\theta, x_i, y_i) \right]\right] \\
\end{align}
$$

* For the finite dataset of size $$N$$ this expectation over each datapoint gradient is nothing but the full dataset gradient and hence we can simplify the above as 

$$
\begin{align}
&\mathbb{E}_{B} \left[\frac{1}{m} \sum_{i=1:m}\left[ \nabla_{\theta} \, \ell(\theta, x_i, y_i) \right]\right] \\
&= \left[\frac{1}{m} \sum_{i=1:m} \mathbb{E}_{x_i, y_i \sim \mathbb{P}(x,y)} \left[ \nabla_{\theta} \, \ell(\theta, x_i, y_i) \right]\right] \\
&= \frac{1}{m} \sum_{i=1:m} \nabla_{\theta} L(\theta)   \\
&= \frac{1}{m} \sum_{i=1:m}\frac{1}{N} \sum_{j=1:N}\left[ \nabla_{\theta} \, \ell(\theta, x_j, y_j) \right] \\
&= \frac{1}{m}.m \frac{1}{N} \sum_{j=1:N}\left[ \nabla_{\theta} \, \ell(\theta, x_j, y_j) \right] \\
&= \frac{1}{N} \sum_{j=1:N}\left[ \nabla_{\theta} \, \ell(\theta, x_j, y_j) \right] = \nabla_{\theta} L(\theta)
\end{align}
$$

* So the gradient used in Stochastic gradient descent is an unbiased Estimator of the full dataset gradient descent. The gradient descent rule for Stochastic Gradient descent is given by
$$
\begin{align}
\theta^{(t+1)} = \theta^{(t)} - \eta_{t}\nabla_{\theta} L_{m}(\theta)
\end{align}
$$
&emsp;where $$t$$ is the iteration number, $$\eta_{t}$$ is the learning rate at iteration $$t$$ while $$L_{m}(\theta)$$ is the mini-batch gradient based on $$m$$ samples. For strict stochastic gradient descent $$m= 1$$.

## SGD connection to Robbins Monroe Algorithm

* Robbins Monroe Algorithm is a class of optimization technique that deals with finding the root $$x^{*} $$ of a equation $$g(x^{*}) = 0$$. When the function $$g(x)$$ is known or accessible we have in our high school maths taken help of Netwon Raphson method where iteratively we reached the solution by applying the below recurrence starting from some $$x_0$$

$$
\begin{align}
x_{n+1} = x_{n} - \frac {g(x_n)}{g'(x_n)} 
\end{align}

 



 






