---
layout: post
title: "Contraction Mapping and Convergences in Machine Learning"
date: 2025-07-02 00:00:00 -0000
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

Example - The sequence $$\{x_n\}$$ where $$x_{n} = \frac{1}{n} $$ is a Cauchy sequence.

   a. To prove the same we would like to find an $$N$$ for every $$\epsilon$$ such that fo every pair $$m,n \ge N $$  the inequality $$\lvert \frac{1}{m} - \frac{1}{n} \rvert \lt \epsilon$$  holds.
   
   b. Applying the modulus inequality we get 
   
   $$ \lvert \frac{1}{m} - \frac{1}{n} \rvert \lt \lvert \frac{1}{m} \rvert + \lvert \frac{1}{n} \rvert$$
   
   c. For a given $$\epsilon$$ we can choose $$N \gt \frac{2}{\epsilon} $$ which will ensure for both positive integers $$m,n \ge N$$ we will have $$\frac{1}{n} \gt \frac{\epsilon}{2}$$ and $$\frac{1}{m} \gt \frac{\epsilon}{2}$$. Hence,
   
   $$ \lvert \frac{1}{m} - \frac{1}{n} \rvert \lt \lvert \frac{1}{m} \rvert + \lvert \frac{1}{n} \rvert \lt \frac{\epsilon}{2} + \frac{\epsilon}{2} = \epsilon $$
      
   d. Since we are able to find $$N \gt \frac{2}{\epsilon} $$ for the chosen $$\epsilon$$ in $$\lvert x_{m} - x_{n} \rvert \lt \epsilon$$ the sequence is a Cauchy sequence.







