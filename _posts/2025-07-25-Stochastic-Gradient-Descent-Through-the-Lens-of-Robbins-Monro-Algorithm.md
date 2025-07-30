---
layout: post
title: "Stochastic Gradient Descent Through the Lens of Robbins Monro Algorithm"
date: 2025-07-23 00:00:00 -0000
author: Santanu Pattanayak
tags: Gradient Descent, Stochastic Gradient Descent, Robbins Monro Algorithm 
---

## Introduction to Stochastic Gradient Descent 

* **Stochastic Gradient Descent** is a variant of Gradient Descent where instead of computing the Gradient of the loss over an entire dataset at all iterations, we compute the gradient of the loss of a **randomly sampled datapoint** in each iteration. When each iteration uses a subset $$m << N$$ where $$N$$ is the number of  datapoints its called a **Minibatch Gradient Descent**.

* If the loss for the datapoint $$(x,y)$$ is $$l(x,y)$$ where $$x$$ is the input feature vector and $$y$$ is the output, then the total expected loss assuming data distribution $$\mathbb{P}(x,y)$$ is as follows:
  
$$
\begin{align}
L(\theta) &= \mathop {\mathbb E}_{x,y \sim P(x,y)} \left[\ell(\theta,x,y)\right] \\
\end{align}
$$

* The gradient of the loss similarly can be expressed as:
$$
\begin{align}
\nabla_{\theta} L(\theta) = \mathbb{E}_{x,y \sim P(x,y)} \left[ \nabla_{\theta} \, \ell(\theta, x, y) \right]
\end{align}
$$

* As stated earlier, in Stochastic gradient descent the entire gradient is approximated by the gradient of the loss objective of a single datapoint. It is obvious that the loss for the single datapoint is an **unbiased estimator** of the loss objective of the actual data distribution gradient as the expectation over the single datapoint loss gradient is equal to the gradient over the entire dataset loss gradient.
 If the gradient of the single datapoint loss is $$\left[ \nabla_{\theta} \, \ell(\theta, x, y) \right]$$ then taking expectation of it over the entire dataset distribution $$\mathbb{P}(x,y) $$ we get

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
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;where $$t$$ is the iteration number, $$\eta_{t}$$ is the learning rate at iteration $$t$$ while $$L_{m}(\theta)$$ is the mini-batch gradient based on $$m$$ samples. For strict stochastic gradient descent $$m= 1$$.

## Robbins Monro Algorithm

* Robbins Monro Algorithm is a class of optimization technique that deals with finding the root $$x^{*} $$ of an equation $$g(x^{*}) = 0$$. When the function $$g(x)$$ is known or accessible we have in our high school taken help of algorithms such as Netwon Raphson method where iteratively we reached the solution by applying the below recurrence starting from some $$x_0$$

$$
\begin{align}
x_{n+1} = x_{n} - \frac {g(x_n)}{g'(x_n)} 
\end{align}
$$ 

* However, not all functions $$g(x)$$ are directly observable or computable and what we can get is some noisy version of $$g(x)$$ say $$\tilde{g(x)} = g(x) + \epsilon$$. If in such cases
$$
\begin{align}
\mathbb{E}_{\epsilon \sim \mathbb{P}(\epsilon)} \left[\tilde{g(x)}\right] = g(x)
\end{align}
$$
&emsp;&emsp;&emsp; then we can iteratively solve for $$x^{*}$$ by the below update rule given by the Morris Monro Algorithm

$$
\begin{align}
x_{t+1} = x_{t} - \eta_{t}\tilde{g}(x_t)  
\end{align}
$$

* To avoid divergence the sequence of the learning rate $$\eta$$ should follow  

$$
\begin{align}
\sum_{t=1:\infty} \eta_{t} = \infty \\
\sum_{t=1:\infty} \eta_{t}^{2} \lt \infty \\
\end{align}
$$ 

## Stochastic Gradient Descent connection to Robbins Monro Algorithm

* In Stochastic Gradient Descent we aim to minimize an expected objective $$L(\theta) = \mathbb{E}_{x,y \sim \mathbb{P}(x,y)}\left[\ell(x,y)\right] $$ by finding the $$\theta^{*}$$ that makes the gradient $$\nabla_{\theta}L(\theta^{*}) = 0$$

* We can think of this minimization problem as a root finding problem to the equation 

$$
\begin{align}
\nabla_{\theta}L(\theta) = 0
\end{align}
$$

* Since the gradient of the loss expectation $$\nabla_{\theta} L(\theta) = \mathbb{E}_{x,y \sim \mathbb{P}(x,y)}\left[\nabla_{\theta}\ell(x,y)\right] $$ or it's finite approximation using a large training dataset of $$N$$ samples $$\frac{1}{N} \sum_{i=1:N}\left[ \nabla_{\theta} \, \ell(\theta, x_i, y_i) \right]$$ is not tractable especially in deep learning network for resource constraints we compute minibatch gradients $$\nabla_{\theta}L_{m}(\theta) $$ which are noisy versions of the expected gradients $$\nabla_{\theta}L(\theta) $$. Also as we have seen before the minibatch gradients are unbiased estimators of full dataset gradients or expected data distribution gradient both of which we have represented by $$\nabla_{\theta}L(\theta) $$. Hence 

$$
\begin{align}
\mathbb{E} \nabla_{\theta}L_{m}(\theta) = \nabla_{\theta}L(\theta)
\end{align}
$$

* The above two points allow us to use Robbins Monro algorithm to solve the optimization problem using approximate gradients given by the mini-batches. The update rule for the same is as follows
$$
\begin{align}
\theta^{(t+1)} = \theta^{(t)}- \eta_{t} \nabla_{\theta} L_{m}(\theta)
\end{align}
$$

* Thus, we see Stochastic Grdient Descent is a practical application of Robbins-Monro stochastic approximation to modern optimization problems.


