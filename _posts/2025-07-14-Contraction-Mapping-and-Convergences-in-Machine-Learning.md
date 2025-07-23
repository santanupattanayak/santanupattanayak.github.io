---
layout: post
title: "Contraction Mapping and Convergences in Machine Learning"
date: 2025-07-14 00:00:00 -0000
author: Santanu Pattanayak
tags: banach fixed point, Contraction Mapping, gradient descent convergence through Contraction Mapping. 
---

## Contraction Mapping 

*  A mapping or transform $$T$$ on a complete metric space $$(X,d)$$ onto itself is said to be a Contraction mapping if $$T$$ brings the points closer to each other. Mathematically, $$T:X \leftarrow X$$ if said to be a contraction  if there exists $$0 \le c \le 1$$ such that

$$ \lVert Tx - Ty \rVert \le c \lVert x-y \rVert \: \forall x,y \in X $$

&nbsp; &nbsp; &nbsp; As illustrated in the Figure 1. we can see that the distance between two points $$x$$ and $$y$$ shrinks on application of the transform
<br>
&nbsp; &nbsp; &nbsp;  $$T$$  and hence $$T$$ is a   contraction.

<p align="center">
<img  width="320" height="320" alt="image" src="https://github.com/user-attachments/assets/55ebd7b5-9e9c-462b-8d63-4d5e4d967ff5" />
<\p>



*Figure 1. Contraction Mapping in a 2D metric space*

  
    





*  When $$c \lt 1 $$ we call it a **strict contraction**. A strict contraction has a unique fixed point $$x$$ that satisfies $$Tx = x$$

* In general the **Contraction Mapping theorem** states that if $$T:X \rightarrow X$$ is a contraction mapping in a **complete metric space** $$(X,d)$$ then there is exactly one fixed point $$x \in X $$ that satisfies $$Tx=x$$.  When the metric space is a **normed vector space**, the Contraction mapping theorem is called **Banach Fixed Point Theorem**. 


We can see in the definition of Contraction Mapping theorem the term **Complete metric space** has come up. We will first discuss about **Cauchy sequences** to help define **Complete metric spaces**  

## Cauchy sequences
* Any seqence $${\{ x_n \}}$$ in a metric space is called a Cauchy sequence if the distance between the consecutive terms get arbitrarily smaller as the sequence progresses.
  Mathematically, if for every positive real number $$\epsilon \gt 0 $$ there exists a positive integer $$N$$ such that for any pair of positive integers $$m,n \ge N$$ the below holds, the sequence $${\{ x_n \}}$$ is said to be Cauchy. 

$$ \lvert x_{m} - x_{n} \rvert \lt \epsilon$$

*  Alternatelty if $$\lim_{m,n\to\infty} \lvert x_{m} - x_{n} \rvert \to 0 $$ the sequence $$\{x_n\}$$ is Cauchy.

**Example**

The sequence $$\{x_n\}$$ where $$x_{n} = \frac{1}{n} $$ is a Cauchy sequence. To prove the same we would like to find  $$N_{\epsilon}$$ for every $$\epsilon \gt 0$$ such that for every pair $$m,n \ge N_{\epsilon} $$  the inequality $$\lvert \frac{1}{m} - \frac{1}{n} \rvert \lt \epsilon$$  holds.  Applying the modulus inequality and the fact that $$\frac{1}{n} < \frac{1}{N_\epsilon}$$ and $$\frac{1}{m} < \frac{1}{N_\epsilon}$$ we get 
   
   $$ \lvert \frac{1}{m} - \frac{1}{n} \rvert \le \lvert \frac{1}{m} \rvert + \lvert \frac{1}{n} \rvert \le \frac{2}{N} \lt \epsilon $$  
   
Based on the above, we can pick $$N_{\epsilon} \gt \frac{2}{\epsilon} $$ to statisfy the Cauchy condition. Hence $$x_{n} = \frac{1}{n}$$ is a Cauchy sequence.

To determine what the sequence converges to, we can compute $$\lim_{n\to\infty} \frac{1}{n} $$ which is $$0$$. Hence $$x_{n} = \frac{1}{n}$$  converges to $$0$$ which is called **limit of the sequence**.

## Complete Metric Space 

Now that we have defined **Cauchy sequences** it would be easy to define the Complete Metric Space. 

* A **Complete Metric Space** is a metric space in which every Cauchy sequence converges to the **limit** of the sequence.  In essence its a space which contains all the limit points of every possible Cauchy sequence in the metric space.
*  Let us look at the convergence of the sequence below in the metric space of rationals $$\mathbb{Q} $$

$$ x_{n} = \left(1 + \frac{1}{n}\right)^{n} ; n \in \mathbb{N} $$

This sequence proceeds as a sequence of rationals 
   
   $$ {2}, \frac{9}{4},\frac{64}{27} ... $$
   
   and finally would have converged to  
   
   $$ \lim_{n\to\infty} x_{n} = \left(1 + \frac{1}{n}\right)^{n}  = e $$
   

   The sequence is Cauchy, however since $$e$$ is not a rational number hence it cannot converge in the metric space of $$\mathbb{Q} $$. In essense $$\mathbb{Q} $$ is not a complete metric space. The sequence would have converged in the metric space $$\mathbb{R} $$ as the limit of the sequence $$e \in \mathbb{R} $$. For a sequence to converge the terms of the sequence have to get arbritarily close to each other as the sequence progresses (it has to be a Cauchy sequence) and the limit of the sequence also needs to be in the metric space. So for convergence we desire a Complete metric space where all Cauchy sequences can converge.

## Revisiting the Contraction Mapping Theorem 

* Now that we know about Cauchy Sequences and Complete metric spaces, we will try to prove the existence of the fixed point satisfying $$Tx = x$$ by showing that the contraction mapping $$T$$ leads to a Cauchy sequence.  The sequence is defined by the iteration $$x_{n+1} = Tx_{n}$$

* If we take the sequence values at $$x_{m}$$ and $$x_{n}$$ for positive integers $$m,n \ge N$$ and use the Contraction mapping property

$$
\begin{align}
\lvert x_{m} - x_{n} \rvert &= \lvert Tx_{m-1} - Tx_{n-1} \rvert \\
&\le c\lvert x_{m-1} - x_{n-1} \rvert \\
&\le c^{n}\lvert x_{m-n} - x_{0} \rvert \\
\end{align}
$$

As $$m,n\to\infty$$ since for the contraction mapping $$c \lt 1$$ hence $$c^{n}\lvert x_{m-n} - x_{0} \rvert \to 0$$ . This proves that the sequence $$\{x_{n}\}$$ generated by $$T$$ is a Cauchy sequence. Since the Contraction Mapping is defined on a Complete metric space the Cauchy sequence should converge to the limit of the sequence which is a unique fixed point.

In the rest of the Chapter we will try to prove the Convergence of iterative alogithms in Machine Learning using the Contraction Mapping Theorem. Specifically we will study the Convergence of Gradient Descent for a well known Convex function as well as the Convergence of the Value Function in Reinformacement learning using Contraction Mapping theorem.

## Gradient Descent Convergence of Linear Regression

* Let's study the convergence of Linear Regression Least square objective $$L = \frac{1}{2}{\lVert {X\theta - Y} \rVert}^{2}$$ using Gradient descent. Here $$X \in \mathbb{R}^{p\times q} $$ is the data matrix of $$p$$ datapoints of dimension $$q$$ while the parameter of Linear regression $$\theta \in \mathbb{R}^{q}$$ is what we want to estimate through the iterative process of Gradient Descent starting from some initial value of $$\theta^{(0)}$$. $$Y \in \mathbb{R}^{p}$$ is the vector containing the targets for the $$p$$ datapoints.

* The **gradient descent parameter update** rule is as follows where $$t$$ is the iteration number:

  
  $$
  \begin{align}
  \theta^{(t+1)} = \theta^{(t)} - \eta \nabla_{\theta} L(\theta^{(t)}) 
  \end{align}
  $$
  
  The gradient of the objective $$L$$ with respect to the parameter vector $$\theta^{(t)}$$ is $$\nabla_{\theta} L(\theta^{(t)}) = X^{T}(X\theta^{(t)} - Y) $$. Substituting the same in the generic gradient descent update rule simplifies the same as follows:

  $$
  \begin{align}
  \theta^{(t+1)} &= \theta^{(t)} - \eta(X^{T}(X\theta^{(t)} - Y)) \\
  &= \theta^{(t)} - \eta(X^{T}X\theta^{(t)} - X^{T}Y)
  \end{align} 
  $$
   
   We can think about gradient descent as an interative operation with $$\theta^{(t+1)} = T\theta^{(t)}$$. Hence the thing to study is if the Gradient descent operator $$T$$ is a contraction mapping.

* Lets look at the the gradient descent operation at iterations $$m$$ and $$n$$ using the L2 norm 

$$
\begin{align}
& \lVert T\theta^{(m)} - T\theta^{(n)} \rVert \\
&=\lVert \theta^{(m)} - \eta(X^{T}X\theta^{(m)} - X^{T}Y) - \theta^{(n)} + \eta(X^{T}X\theta^{(n)} - X^{T}Y) \rVert \\
&= \lVert (\theta^{(m)} - \theta^{(n)}) - \eta(X^{T}X\theta^{(m)} - X^{T}X\theta^{(n)}) \rVert \\
&= \lVert (I - \eta X^{T}X) (\theta^{(m)} - \theta^{(n)}) \rVert  \\
&\le \lVert (I - \eta X^{T}X)\rVert \lVert(\theta^{(m)} - \theta^{(n)}) \rVert
\end{align}
$$

* So Gradient descent for least squares would be a contraction mapping if $$\lVert (I - \eta X^{T}X)\rVert \lt 1$$ . $$X^{T}*X$$ being a positive semidefinite symmetric matrix has eigen values $$\lambda_{i} \ge 0$$. The norm of the $$\lVert (I - \eta X^{T}X)\rVert$$ is nothing but
  
$$ \max_{i} \lvert 1 - \eta\lambda_{i} \rvert$$
  
* This norm should be less than 1 for gradient descent to be a contraction and subsequently guarantee convergence. Given that any of the eigen values of $$X^{T}*X$$  can be associated   with the norm of the  $$\lVert (I - \eta X^{T}X)\rVert$$ it can be seen that the tightest bound of the learning rate is provided by the maximum eigen value as shown below

$$
\begin{align}
 &    \lvert 1 - \eta\lambda_{max} \rvert < 1            \\
 &  \Rightarrow -1 \lt 1 - \eta\lambda_{max} < 1  \\
 &  \Rightarrow 0 \lt \eta \lt \frac{2}{\lambda_{max}}   
 \end{align}
$$

## Value function convergence under the Bellman operator

* In this section we will look at the **Value function convergence** under a given policy $$\pi$$ using the Supremum norm(max norm) over the state space. We will use the standard notations of $$s$$ and $$s^{'}$$ for current and next state, $$a$$ for action at current state $$s$$. We denote the immediate reward at state $s$ on taking action $$a$$ as per the policy $$\pi$$ as $$r(s,a)$$ and $$\gamma$$ as the discount factor such that $$ 0 \le \gamma < 1 $$.

* We denote the value function at a given state by $$V(s)$$ while the overall Value function for all states as $$V$$

* As per the **Bellman operator**, which we denote by $$T$$ here we have the following recurrence wrt to Value function

  $$TV(s) = \max_{a} (r(s,a) + \gamma \sum_{s'} \mathbb{P}(s^{'} | s,a) V(s')) $$

* Lets take a lookat the Value function for a given state $$s$$ under the same policy $$\pi$$ at two iteration number $$m$$ and $$n$$ and compute their different

 $$ \lvert TV^{(m)}(s) - TV^{(n)}(s) \rvert$$
 $$= \lvert \max_{a} (r(s,a) + \gamma \sum_{s'} \mathbb{P}(s^{'} | s,a) V^{(m)}(s'))$$
 $$ - \max_{a} (r(s,a) + \gamma \sum_{s'} \mathbb{P}(s^{'} | s,a) V^{(n)}(s')) \rvert$$      

* The immediate reward cancels out and the expression simplifies to

 $$ \lvert TV^{(m)}(s) - TV^{(n)}(s) \rvert = \lvert \max_{a} \gamma \sum_{s'} \mathbb{P}(s^{'} | s,a) (V^{(m)}(s') - V^{(n)}(s') )   \rvert $$ 

* Since $$\lvert \max_a (x_a - y_a) \rvert  \le \max_a\lvert x_a - y_a \rvert $$ we can convert our equality equation into an inequality one as below

 $$ \lvert TV^{(m)}(s) - TV^{(n)}(s) \rvert \le  \max_{a} \gamma \lvert \sum_{s'} \mathbb{P}(s^{'} | s,a) (V^{(m)}(s') -  V^{(n)}(s') )  \rvert $$ 

* Applying triangle inequality to the weighted sum inside the abs value we get
  
 $$ \lvert TV^{(m)}(s) - TV^{(n)}(s) \rvert \le  \max_{a} \gamma \sum_{s'} \mathbb{P}(s^{'} | s,a) \lvert V^{(m)}(s') - V^{(n)}(s')    \rvert $$ 

* The fact that $$\lvert V^{(m)}(s') - V^{(n)}(s')    \rvert $$ is no greater than the supremum norm(maximum across all states) the Bellaman operator inequality simplifies to

 $$ \lvert TV^{(m)}(s) - TV^{(n)}(s) \rvert \le  \max_{a} \gamma \sum_{s'} \mathbb{P}(s^{'} | s,a) {\lVert V^{(m)} - V^{(n)} \rVert}_{\infty} $$

 * The sum of the probability across all states is 1 and hence the inequality further simplies to 
 
 $$ \lvert TV^{(m)}(s) - TV^{(n)}(s) \rvert \le   \gamma  {\lVert V^{(m)} - V^{(n)} \rVert}_{\infty} $$

* Finally since $$\lvert TV^{(m)}(s) - TV^{(n)}(s) \rvert \le   \gamma  {\lVert V^{(m)} - V^{(n)} \rVert}_{\infty} $$ for any state $$s$$ it should be true for the state $$s$$ corresponding to the maximum value of $$\lvert TV^{(m)}(s) - TV^{(n)}(s) \rvert$$ which is nothing but the supremum norm of $$TV^{(m)} - TV^{(n)} $$ . Hence the inequality takes the final form as below

 $$ \lVert TV^{(m)} - TV^{(n)} \rVert_{\infty} \le   \gamma  {\lVert V^{(m)} - V^{(n)} \rVert}_{\infty} $$

* Since the discount factor $$0\le \gamma \le 1 $$ hence the Bellman operator is a contraction mapping and the Value function converges.



