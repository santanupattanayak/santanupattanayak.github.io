---
layout: post
title: "Contraction Mapping and Convergences in Machine Learning"
date: 2025-07-14 00:00:00 -0000
author: Santanu Pattanayak
tags: banach fixed point, Contraction Mapping, gradient descent convergence through Contraction Mapping. 
---

## Contraction Mapping 

1.  A mapping or transform $$T$$ on a complete metric space $$(X,d)$$ onto itself is said to be a Contraction mapping if $$T$$ brings the points closer to each other. Mathematically, $$T:X \leftarrow X$$ if said to be a contraction mapping if there exists $$0 \le c \le 1$$ such that

$$ \lVert Tx - Ty \rVert \le c \lVert x-y \rVert \: \forall x,y \in X $$


2.  When $$c \lt 1 $$ we call it a strict contraction. A strict contraction has a unique fixed point $$x$$ that satisfies $$Tx = x$$

3. In general the **Contraction Mapping theorem** states that if $$T:X \rightarrow X$$ is a contraction mapping in a **complete metric space** $$(X,d)$$ then there is exactly one fixed point $$x \in X $$ that satisfies $$Tx=x$$

We can see in the definition of Contraction Mapping theorem the term **Complete metric space** has come up. We will first discuss about **Cauchy sequences** to help define **Complete metric spaces**  

## Cauchy sequences
1. Any seqence $${\{ x_n \}}$$ in a metric space is called a Cauchy sequence if the distance between the consecutive terms get arbitrarily closer as the sequence progresses.
2.  Mathematically, if for every positive real number $$\epsilon \gt 0 $$ there exists a positive integer $$N$$ such that for any pair of positive integers $$m,n \ge N$$ the below holds, the sequence $${\{ x_n \}}$$ is said to be Cauchy. 

$$ \lvert x_{m} - x_{n} \rvert \lt \epsilon$$

**Example**

The sequence $$\{x_n\}$$ where $$x_{n} = \frac{1}{n} $$ is a Cauchy sequence. To prove the same we would like to find  $$N_{\epsilon}$$ for every $$\epsilon \gt 0$$ such that for every pair $$m,n \ge N_{\epsilon} $$  the inequality $$\lvert \frac{1}{m} - \frac{1}{n} \rvert \lt \epsilon$$  holds.  Applying the modulus inequality and the fact that $$\frac{1}{n} < \frac{1}{N_\epsilon}$$ and $$\frac{1}{m} < \frac{1}{N_\epsilon}$$ we get 
   
   $$ \lvert \frac{1}{m} - \frac{1}{n} \rvert \le \lvert \frac{1}{m} \rvert + \lvert \frac{1}{n} \rvert \le \frac{2}{N} \lt \epsilon $$  
   
Based on the above, we can pick $$N_{\epsilon} \gt \frac{2}{\epsilon} $$ to statisfy the Cauchy condition. Hence $$x_{n} = \frac{1}{n}$$ is a Cauchy sequence.

To determine what the sequence converges we can compute $$lim_{n \to \infty} \frac{1}{n} $$ which is $$0$$. Hence $$x_{n} = \frac{1}{n}$$  converges to $$0$$ which is called limit of the sequence.

## Complete Metric Space 

Now that we have defined **Cauchy sequences** it would be easy to define the Complete Metric Space. 

1. A **Complete Metric Space** is a metric space in which every Cauchy sequence converges to a point in the space called the **limit** of the sequence.  In essence its a space which contains all the limit points of every possible Cauchy sequence in the metric space.
2. Let's us look at the convergence of the sequence below in the metric space of rationals $$\mathbb{Q} $$

$$ x_{n} = \left(1 + \frac{1}{n}\right)^{n} ; n \in \mathbb{N} $$

This sequence proceeds as a sequence of rationals 
   
   $$ {2}, \frac{9}{4},\frac{64}{27} ... $$
   
   and finally would have converged to  
   
   $$ \lim_{n\to\infty} x_{n} = \left(1 + \frac{1}{n}\right)^{n}  = e $$
   

   The sequence is Cauchy, however since $$e$$ is not a rational number hence it cannot converge in the metric space of $$\mathbb{Q} $$. In essense $$\mathbb{Q} $$ is not a complete metric space. The sequence would have converged in the metric space $$\mathbb{R} $$ as the limit of the sequence $$e \in \mathbb{R} $$. So for a sequence to converge the terms of the sequence have to get arbritarily close to each other as the sequence progresses (Cauchy sequence) and the limit of the sequence also needs to be in the metric space. So for convergence we desire a Complete metric space where all Cauchy sequences can converge.

   








