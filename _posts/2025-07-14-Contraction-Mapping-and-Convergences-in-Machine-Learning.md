---
layout: post
title: "Contraction Mapping and Convergences in Machine Learning"
date: 2025-07-14 00:00:00 -0000
author: Santanu Pattanayak
tags: banach fixed point, Contraction Mapping, gradient descent convergence through Contraction Mapping. 
---

## Contraction Mapping 

*  A mapping or transform $$T$$ on a complete metric space $$(X,d)$$ onto itself is said to be a Contraction mapping if $$T$$ brings the points closer to each other. Mathematically, $$T:X \leftarrow X$$ if said to be a contraction mapping if there exists $$0 \le c \le 1$$ such that

$$ \lVert Tx - Ty \rVert \le c \lVert x-y \rVert \: \forall x,y \in X $$


*  When $$c \lt 1 $$ we call it a strict contraction. A strict contraction has a unique fixed point $$x$$ that satisfies $$Tx = x$$

* In general the **Contraction Mapping theorem** states that if $$T:X \rightarrow X$$ is a contraction mapping in a **complete metric space** $$(X,d)$$ then there is exactly one fixed point $$x \in X $$ that satisfies $$Tx=x$$


We can see in the definition of Contraction Mapping theorem the term **Complete metric space** has come up. We will first discuss about **Cauchy sequences** to help define **Complete metric spaces**  

## Cauchy sequences
* Any seqence $${\{ x_n \}}$$ in a metric space is called a Cauchy sequence if the distance between the consecutive terms get arbitrarily closer as the sequence progresses.
  Mathematically, if for every positive real number $$\epsilon \gt 0 $$ there exists a positive integer $$N$$ such that for any pair of positive integers $$m,n \ge N$$ the below holds, the sequence $${\{ x_n \}}$$ is said to be Cauchy. 

$$ \lvert x_{m} - x_{n} \rvert \lt \epsilon$$

*  Alternatelty if $$\lim_{m,n\to\infty} \lvert x_{m} - x_{n} \rvert \to 0 $$ the sequence $$\{x_n\}$$ is Cauchy.

**Example**

The sequence $$\{x_n\}$$ where $$x_{n} = \frac{1}{n} $$ is a Cauchy sequence. To prove the same we would like to find  $$N_{\epsilon}$$ for every $$\epsilon \gt 0$$ such that for every pair $$m,n \ge N_{\epsilon} $$  the inequality $$\lvert \frac{1}{m} - \frac{1}{n} \rvert \lt \epsilon$$  holds.  Applying the modulus inequality and the fact that $$\frac{1}{n} < \frac{1}{N_\epsilon}$$ and $$\frac{1}{m} < \frac{1}{N_\epsilon}$$ we get 
   
   $$ \lvert \frac{1}{m} - \frac{1}{n} \rvert \le \lvert \frac{1}{m} \rvert + \lvert \frac{1}{n} \rvert \le \frac{2}{N} \lt \epsilon $$  
   
Based on the above, we can pick $$N_{\epsilon} \gt \frac{2}{\epsilon} $$ to statisfy the Cauchy condition. Hence $$x_{n} = \frac{1}{n}$$ is a Cauchy sequence.

To determine what the sequence converges we can compute $$\lim_{n\to\infty} \frac{1}{n} $$ which is $$0$$. Hence $$x_{n} = \frac{1}{n}$$  converges to $$0$$ which is called limit of the sequence.

## Complete Metric Space 

Now that we have defined **Cauchy sequences** it would be easy to define the Complete Metric Space. 

* A **Complete Metric Space** is a metric space in which every Cauchy sequence converges to a point in the space called the **limit** of the sequence.  In essence its a space which contains all the limit points of every possible Cauchy sequence in the metric space.
*  Let's us look at the convergence of the sequence below in the metric space of rationals $$\mathbb{Q} $$

$$ x_{n} = \left(1 + \frac{1}{n}\right)^{n} ; n \in \mathbb{N} $$

This sequence proceeds as a sequence of rationals 
   
   $$ {2}, \frac{9}{4},\frac{64}{27} ... $$
   
   and finally would have converged to  
   
   $$ \lim_{n\to\infty} x_{n} = \left(1 + \frac{1}{n}\right)^{n}  = e $$
   

   The sequence is Cauchy, however since $$e$$ is not a rational number hence it cannot converge in the metric space of $$\mathbb{Q} $$. In essense $$\mathbb{Q} $$ is not a complete metric space. The sequence would have converged in the metric space $$\mathbb{R} $$ as the limit of the sequence $$e \in \mathbb{R} $$. So for a sequence to converge the terms of the sequence have to get arbritarily close to each other as the sequence progresses (Cauchy sequence) and the limit of the sequence also needs to be in the metric space. So for convergence we desire a Complete metric space where all Cauchy sequences can converge.

## Revisiting the Contraction Mapping Theorem 

* Now that we know about Cauchy Sequences and Complete metric spaces, we will try to prove the existence of the fixed point satisfying $$Tx = x$$ by showing that the contraction mapping $$T$$ leads to a Cauchy sequence.  The sequence is defined by the iteration $$x_{n+1} = Tx_{n}$$

* If we take the sequence values at $$x_{m}$$ and $$x_{n}$$ for positive integers $$m,n \ge N$$ and use the Contraction mapping property

$$ \lvert x_{m} - x_{n} \rvert = \lvert Tx_{m-1} - Tx_{n-1} \rvert $$
                               $$\le c\lvert x_{m-1} - x_{n-1} \rvert$$ 
                               $$\le c^{n}\lvert x_{m-n} - x_{0} \rvert$$ 

As $$m,n\to\infty$$ since for the contraction mapping $$c \lt 1$$ hence $$c^{n}\lvert x_{m-n} - x_{0} \rvert \to 0$$ . This proves that the sequence $$\{x_{n}\}$$ generated by $$T$$ is a Cauchy sequence. Since the Contraction Mapping is defined on a Complete metric space the Cauchy sequence should converge to the limit of the sequence which is a unique fixed point.

In the rest of the Chapter we will try to prove the Convergence of iterative alogithms in Machine Learning using the Contraction Mapping Theorem. Specifically we will study the Convergence of Gradient Descent for Convex functions as well as the Convergence of the Value Function in Reinformacement learning using Contraction Mapping theorem.

## Gradient Descent Convergence of Linear Regression

* Let's study the convergence of Linear Regression Least square objective $$L = \frac{1}{2}{\lVert {X\theta - Y} \rVert}^{2}$$ using Gradient descent. Here $$X \in \mathbb{R}^{m\times n} $$ is the data matrix of $$m$$ datapoints of dimension $$n$$ while the parameter of Linear regression $$\theta \in \mathbb{R}^{n}$$ is what we want to estimate through the iterative process of Gradient Descent starting from some initial value of $$\theta^{(0)}$$. $$Y \in \mathbb{R}^{m}$$ is the vector containing the targets for the $$m$$ datapoints.

* The **gradient descent parameter update** rule is as follows where $$t$$ is the interation number:
     $$\theta^{(t+1)} = \theta^{(t)} - \eta \nabla_{\theta} L(\theta^{(t)})$$ .The gradient of the objective $$L$$ with repsect to the parameter vector $$\theta^{(t)}$$ is $$\nabla_{\theta} L(\theta^{(t)}) = X^{T}(X\theta^{(t)} - Y) $$. Substituting the same in the generic gradient descent update rule simplifies the same to below

   $$\theta^{(t+1)} = \theta^{(t)} - \eta(X^{T}(X\theta^{(t)} - Y)) = \theta^{(t)} - \eta(X^{T}X\theta^{(t)} - X^{T}Y)$$
   
   We can think about gradient descent as an interative operation with $$\theta^{(t+1)} = T\theta^{(t)}$$. We would like to see if the Gradient descent operator $$T$$ is a contraction mapping.

* Lets look at the the gradient descent operstion at iterations $$m$$ and $$n$$

   $$\lVert T\theta^{(m)} - T\theta^{(n)} \rVert$$
   
   $$=\lVert \theta^{(m)} - \eta(X^{T}X\theta^{(m)} - X^{T}Y) - \theta^{(n)} + \eta(X^{T}X\theta^{(n)} - X^{T}Y) \rVert$$
   
   $$= \lVert (\theta^{(m)} - \theta^{(n)}) - \eta(X^{T}X\theta^{(m)} - X^{T}X\theta^{(n)}) \rVert  $$
   
   $$= \lVert (I - \eta X^{T}X) (\theta^{(m)} - \theta^{(n)}) \rVert  $$
   
   $$\le \lVert (I - \eta X^{T}X)\rVert \lVert(\theta^{(m)} - \theta^{(n)}) \rVert  $$

* So Gradient descent for least squares would be a contraction mapping if $$\lVert (I - \eta X^{T}X)\rVert \lt 1$$ . $$X^{T}*X$$ being a positive semidefinite symmetric matrix has eigen values $$\lambda_{i} \ge 0$$. The norm of the $$\lVert (I - \eta X^{T}X)\rVert$$ is nothing but

$$ \max_{i} \lvert 1 - \eta\lambda_{i} \rvert$$

and this norm should be less than 1 for contraction mapping and subsequent convergence of the gradient descent.







   

   

    
         

    





   








