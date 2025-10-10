---
layout: post
title: "Universal Approximation theorem"
date: 2025-08-09 00:00:00 -0000
author: Santanu Pattanayak
tags: UAT, Universal approximation theorem, Neural networks, 
---

# Table of Contents
1. [Introduction to Universal Approximation theorem](#introduction)
2. [The Early Versions of Universal Approximation Theorem](#uat)
3. [What guarantees Universal Approximation Theorem](#guat)
4. [Universal Approximation theorem for NN with ReLU activation](#reluaut)

## Introduction <a name="introduction"></a>

The Universal Approximation Theorem (UAT) is one of the cornerstones of modern deep learning theory. At its heart, it asserts that a sufficiently large neural network can approximate any continuous function to arbitrary accuracy under certain conditions. While the statement is popular in machine learning circles, its mathematical foundations are deeply rooted in **real analysis** and **functional analysis**.
This post explores the UAT, its formal statement, and the real analysis theorems that underlie it.

## Norm of a function
The norm of a function would appear multiple times in our discussion of UAT and hence it makes sense to go over it before we dive into the various aspects of UAT.
The Euclidean or the $$l_2$$ norm of a vector $$x$$ in a d-dimensional vector space is defined as  :

$$
\begin{align}
\|x\|_{2} =   (\sum_{i=1}^{d} |x|^{2})^{\frac{1}{2}} 
\end{align}
$$

The $$l_p$$ norm generalizes the $$l_2$$ to any value $$p$$ and is similarly defined as  :
$$
\begin{align}
\|x\|_{p} =   (\sum_{i=1}^{d} |x|^{p})^{\frac{1}{p}} 
\end{align}
$$

$$L^{p}$$ norm is the extension of the $$l_p$$ norm of vectors to functions over a defined domain . Any function $$f(x)$$ can be viewed as a vector of values over the different values of input $$x$$ over a defined domain $$D$$ . The $$L^{p}$$ norm of a function $$f(x); x \in D$$ is given by:  

$$
\begin{align}
\|f\|_{p} = (\int_{x \sim D} |f(x)|^{p} dx)^{\frac{1}{p}}
\end{align}
$$  

When $$p=\infty$$ we get the $$L^{\infty}$$ norm which is also called the Supremum norm. Supremum norm is the nothing but the maximum of the absolute values of the function since:  

$$
\begin{align}
\lim_{p \rightarrow \infty} \|f\|_{p} = \|f\|_{\infty} =  \lim_{p \rightarrow \infty} ( \int_{x \sim D} |f(x)|^{p} dx )^{\frac{1}{p}} = \max_{x} |f(x)| 
\end{align}
$$ 


## The Early Versions of Universal Approximation Theorem <a name="uat"></a>

The earliest rigorous versions of UAT were proved independently by Cybenko (1989) and Hornik, Stinchcombe, and White (1989). One simplified version is:

Let $$\sigma: \mathbb{R} \rightarrow \mathbb{R} $$ be any continuous squashing function such as sigmoid. Then for any continuous function $$f$$ defined on the cube $$[0,1]^{n}$$ and for any tolerance $$\epsilon \gt 0$$ there exists a neural network of the form

$$
\begin{align}
F(x) = \sum_{i=1:m} \alpha_{i}\sigma(w_{i}.x + b_{i})
\end{align}
$$

such that the error in approximation of the function is bounded within $$\epsilon$$ in supremum norm as shown below


$$
\begin{align}
\sup_{x}\left|f(x) - F(x)\right| \lt \epsilon
\end{align}
$$  

The theorem is basically saying that **given enough neurons** we can approximate the function $$f(x)$$ with its Neural Network approximation $$F(x)$$ such that the maximum absolute difference between $$f(x)$$ and $$F(x)$$ can be bounded to arbitrary precision threshold $$\epsilon$$ .



## What guarantees Universal Approximation Theorem <a name="guat"></a>

We can define **pulse functions** within a given range through scaled and shifted sigmoid functions. For example we can define a pulse function in the range $$[a,b]$$ through the subtraction of two scaled sigmoid functions shown below:

$$
\begin{align}
G(x,a,b) = \sigma(k(x -a)) - \sigma(k(x -b)); b > a 
\end{align}
$$  

The scale $$k$$ can be chosen appropriately large. In the below plot we can see three pulses $$y_1 = G(x,a=0.5,b=1.5) $$,  $$y_2 = G(x,a=2,b=3) $$ and $$y_3 = G(x,a=3.2,b=4.2) $$

<img width="1000" height="600" alt="impulse functions" src="https://github.com/user-attachments/assets/f2d5ae59-fff9-4a27-bbf3-4a4233347b4b" />

Now any complicated continuous function can be approximated closely by summing up these pulse functions $$G(x,a,b)$$ which themselves are nothing but composed of sigmoids. Hence the theorem of Universal Approximation theorem as stated in the earlier section holds true where the hidden layer can be composed of the sigmoids that make up the pulse functions and then the output layer weights combine them to form the final function.

If we plan to approximate a function such as below 

<img width="827" height="497" alt="image" src="https://github.com/user-attachments/assets/cef5aa9a-9761-4fe3-9f6e-7150221b6175" />

The same can be composed exactly by combining the earlier illustrated pulse functions as $$y = y_1 + 2y_2 + 1.5y_3$$

<img width="852" height="495" alt="image" src="https://github.com/user-attachments/assets/7cc9b915-87cc-449d-92c2-25a9a974bb43" />

Hence, given only a hidden layer of sigmoid activations we can approximate any complex function with enough neurons in the hidden layer.
Do note that this approximation theorem, doesn't really tell how many neurons are required in the hidden layer or if the learning methodology would be able to learn the approximation well enough.





## Mathematical approach to proving UAT 

We need to define few things to prove the UAT mathematically.

### Linear functional



Let $$( (X, \| \cdot \|)$$ be a **normed linear space** over the field  $$\mathbb{F}$$, where $$\mathbb{F} = \mathbb{R}$$ or $$\mathbb{C}$$.

A **linear functional** on $$X$$ is a map  
$$
f : X \to \mathbb{F}
$$
that satisfies the following two properties for all $$x, y \in X $$ and all scalars $$ \alpha \in \mathbb{F}$$:

$$
\begin{aligned}
f(x + y) &= f(x) + f(y), \\
f(\alpha x) &= \alpha f(x).
\end{aligned}
$$

Thus, a linear functional is simply a **linear map from a vector space into its underlying scalar field**.

---

### Continuity (Boundedness)

A linear functional \( f \) is said to be **continuous** (or **bounded**) if there exists a constant $$C > 0$$ such that  
$$
|f(x)| \le C \|x\|, \quad \forall x \in X
$$

The **operator norm** of $$f$$ is defined as  
$$
\|f\| = \sup_{\|x\| \le 1} |f(x)|
$$

---

### Examples

1. **Finite-dimensional example**

   On $$ X = \mathbb{R}^n $$, define  
   $$
   f(x_1, x_2, \ldots, x_n) = a_1 x_1 + a_2 x_2 + \cdots + a_n x_n,
   $$
   where $$ a_1, \ldots, a_n \in \mathbb{R} $$ are fixed.  
   Then $$ f $$ is a continuous linear functional with  
   $$
   \|f\| = \sqrt{a_1^2 + \cdots + a_n^2} \quad \text{if } \|x\| = \sqrt{x_1^2 + \cdots + x_n^2}
   $$

2. **Functional on $$C[a,b]$$

   Let $$X = C[a,b]$$ with the sup norm  
   $$
   \|g\|_\infty = \sup_{x \in [a,b]} |g(x)|.
   $$
   Define  
   $$
   L(g) = \int_a^b g(x)\,dx
   $$
   Then $$L$$ is a linear functional and is bounded since  
   $$
   |L(g)| \le (b - a)\|g\|_\infty.
   $$



### Remarks

- The collection of all **bounded linear functionals** on \( X \) forms a normed linear space called the **dual space**, denoted \( X^* \).
- Fundamental results like the **Hahn–Banach Theorem** and the **Riesz Representation Theorem** describe the structure of such functionals.








## Does Neural networks with Sigmoid and Tanh activation satisfy Stone-Weierstrass Theorem

Let us now see if Neural Networks with *Sigmoid* or *Tanh* functions would satisfy the requirements of the Stone-Weirstrass theorem.  

- Separate Points: *Sigmoid* and *tanh* are strictly monotone. So for any two points x,y \in  $$C([a,b]^{n})$$ we can find hyperplane $$wz + b$$ such that  
  $$ \sigma(wx +b) \ne \sigma(wy +b) $$
- Constants: By choosing  $$ w=0, \sigma(b) $$ is a constant function (e.g., $$\sigma(0)= \frac{1}{2}$$ for sigmoid while $$\tanh(0) = 0 $$. With scaling and shifting, we can approximate any constant.
- Algebra :
  A neural network with sigmoid functions can approximate a step function. A function  
  $$ G(x) = \sigma(k(x - a)) - \sigma(k(x-b)) $$  

  for a large $$k$$ can approximate a rectangular pulse function in the interval $$[a,b]$$. Any continuous function on a compact (closed and bounded) set can be approximated to any desired accuracy by a sum of rectangular pulse functions, or a "staircase" function. Since shifted and scaled sigmoid functions we approximate pulse  it can approximate a continuous function by adding up the outputs of multiple neurons, each creating a pulse.


## Weierstrass Approximation Theorem and Stone–Weierstrass Theorem

As per **Weierstrass Approximation Theorem**  any continuous function $$f$$ on a compact interval $$[a,b]$$ can be uniformly approximated by a polynomial $$P(x)$$. Mathematically for every function $$f(x) \in C[a,b]$$ there exists a polynomial function P(x) such that

$$
\begin{align}
\sup_{x \in [a,b]} |f(x) - P(x)| \lt \epsilon
\end{align}
$$

Before neural nets, polynomials were the universal approximators. As an example $$sin(x)$$ can be approximated by its Taylor series polynomial expansion. While useful as a start Weierstrass Approximation Theorem is not very useful from a Neural Network perspective as we don't really have explicit polynomial activations in Neural Networks.
hence we move on to **Stone–Weierstrass Theorem** which is a generalization of Weierstrass Approximation Theorem.

**Stone–Weierstrass Theorem** states that  :
Suppose we have a family of function $$A \subset  C([a,b]^{n}) $$ such that:
1. $$A$$ forms an algebra. An algebra satisfies all conditions of vector space with an additional constraint that it is closed under function multiplication.  

   For an algebra if $$f,g \in A$$ then  $$f+g \in A$$ , $$\alpha f \in A$$ and $$fg \in A$$

2. Separate points i.e. there exists atleast one function $$f$$ in $$A$$ for which $$f(x) \ne f(y)$$ if $$x \ne y$$
3. Contains constant functions

Then, $$A$$ is said to be dense in $$C([a,b]^{n})$$
## Universal Approximation theorem for NN with ReLU activation <a name="reluuat"><\a>

The big question is how do we know neural nets can approximate any function?  

This is where real analysis theorems come in. Given any domain  say $$D = [0,1]^{n}$$ we need to find a family of functions which can approximate any continuous function over the domain $$D$$. Generally such a **function space** of continuous functions is denoted by $$C([0,1]^{n})$$.  

When we talk about spaces of real $$\mathbb{R}$$ and rational numbers $$\mathbb{Q}$$ we say $$\mathbb{Q}$$ is **dense** in $$\mathbb{R}$$ since any real number is arbitrarily close to some rational number. In essence if we draw an open set of any non-zero radius $$r$$ around any real number $$x$$ denoted by $$(x-r,x+r)$$ it is bound to contain one or more rational numbers. 
And hence computers which store everything in rational numbers can represent any real number with negligible error.

We need Neural Networks to be able to form a family of functions $$F \subset C([0,1]^{n})$$ which can approximate any function in $$C([0,1]^{n})$$ and hence $$F$$ should be dense in  $$C([0,1]^{n})$$. 

Theorems from Real Analysis guarantees that the family of function Neural networks forms with various activation functions do create dense subset of the desired function space. The two such theorems we will look into are **Weierstrass Approximation Theorem** and its generalization **Stone–Weierstrass Theorem**
