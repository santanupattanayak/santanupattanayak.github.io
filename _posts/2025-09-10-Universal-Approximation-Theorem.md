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

### Linear Functional

Let $$ (X, \| \cdot \|) $$ be a **normed linear space** over the field $$ \mathbb{F} $$, where $$ \mathbb{F} = \mathbb{R} $$ or $$ \mathbb{C} $$.

A **linear functional** on $$ X $$ is a map $$ L : X \to \mathbb{F} $$ satisfying, for all $$ x, y \in X $$ and all scalars $$ \alpha \in \mathbb{F} $$:

$$
\begin{aligned}
L(x + y) &= L(x) + L(y), \\
L(\alpha x) &= \alpha L(x).
\end{aligned}
$$

Thus, a linear functional is simply a **linear map** from a vector space into its underlying scalar field.



### Linear Functionals in Finite-Dimensional Spaces

In a **finite-dimensional normed linear space** such as $$\mathbb{R}^n$$, which is also a **Hilbert space**, a **linear functional** acts on vectors to produce scalars in $$\mathbb{R}$$.  
For any linear functional $$L$$ acting on a vector $$x$$, there exists a corresponding vector $$y$$ in the same space such that  

$$
L(x) = y^{T}x = \langle x, y \rangle
$$

where $$\langle \cdot , \cdot \rangle$$ denotes the inner product.  
The vector $$y$$ belongs to the **dual space** $$X^*$$, which is the space of all bounded linear functionals on $$X$$.  

This correspondence between $$L$$ and $$y$$ is formalized by the **Riesz Representation Theorem**, which asserts that every continuous linear functional on a Hilbert space can be uniquely represented as an inner product with a fixed element of that space.  

Since in finite dimensional Hilbert Space the dual space coincides with the original space, i.e. $$X^* = X$$, any linear functional can be represented as the inner product with some vector in $$X$$. Thus, the dual space of a finite-dimensional Hilbert space is isomorphic to the space itself.

If $$U$$ is a **closed convex subset** of $$X$$, it has an associated orthogonal complement $$U^\perp$$ consisting of all elements orthogonal to $$U$$.  
Any element $$y \in U^\perp$$ satisfies  

$$
\langle x, y \rangle = 0 \quad \text{for all } x \in U
$$

Since each functional $$L$$ corresponds to a unique $$y$$, this also implies that any functional $$L \in X^*$$ satisfies $$L(x) = 0$$ for all $$x \in C$$ whenever $$y \in U^\perp$$.


### Linear Functionals in Infinite-Dimensional Spaces

When we move to an **infinite-dimensional normed linear space**, such as $$C[a,b]$$, the space of continuous functions on a closed and compact interval $$[a,b]$$, the structure of linear functionals changes significantly.  

The space $$C[a,b]$$ is equipped with the **supremum norm**, defined by  

$$
\|f\|_\infty = \sup_{x \in [a,b]} |f(x)|.
$$

Unlike finite-dimensional Hilbert spaces, $$C[a,b]$$ with supremum norm is **not self-dual**.  
Its dual space $$C[a,b]^*$$ consists not of continuous functions, but of more general objects—specifically, *bounded finitely additive signed measures* on $$[a,b]$$.  

This distinction means that the elements of $$C[a,b]^*$$ cannot be represented by inner products with other functions in $$C[a,b]$$; instead, they are expressed in terms of integrals with respect to signed measures.


### Examples of Functionals on $$C[a,b]$$

Consider $$X = C[a,b]$$ with the sup norm $$\|f\|_\infty$$.  
A simple example of a linear functional is  

$$
L(f) = \int_a^b f(x)\,dx.
$$

This functional is linear and bounded since  

$$
|L(f)| \le (b - a)\|f\|_\infty.
$$

Another important example is the **evaluation functional**, which maps each function to its value at a fixed point $$x_0 \in [a,b]$$:  

$$
L_{x_0}(f) = f(x_0).
$$

A slightly modified version of this is the **difference functional**, defined for two fixed points $$x_1, x_2 \in [a,b]$$ as  

$$
L(f) = f(x_1) - f(x_2).
$$


### Representation via Signed Measures

The **Riesz Representation Theorem** for $$C[a,b]$$ states that every continuous linear functional $$L$$ on $$C[a,b]$$ can be represented as an integral with respect to a **finitely additive signed measure** $$\mu$$:  

$$
L(f) = \int_a^b f(x)\, d\mu(x).
$$

Here, measures provide a generalization of length, area, or volume, assigning a (possibly signed) “weight” to subsets of $$[a,b]$$.  

For instance:
- The functional $$L_{x_0}(f) = f(x_0)$$ corresponds to the **Dirac measure** $$\nu = \delta_{x_0}$$, so  
  $$
  L_{x_0}(f) = \int_a^b f(x)\, d\delta_{x_0}(x)
  $$
- The difference functional $$L(f) = f(x_1) - f(x_2)$$ corresponds to the **signed measure** $$\nu = \delta_{x_1} - \delta_{x_2}$$



### Orthogonality and Duality in Infinite Dimensions

In $$\mathbb{R}^n$$, orthogonality has a clear **geometric interpretation**: two vectors are orthogonal if their inner product is zero.  

In infinite-dimensional spaces such as $$C[a,b]$$, this geometric notion no longer applies directly. Instead, orthogonality is defined in terms of the **annihilation** of functionals on subspaces.  

If $$M$$ is a closed subspace of $$C[a,b]$$, the **orthogonal complement** $$M^\perp \subset C[a,b]^*$$ consists of all functionals $$L$$ in the dual space such that  

$$
L(f) = 0 \quad \text{for all } f \in M.
$$

In this sense, the orthogonality in $$C[a,b]$$ is not geometric but **functional**: a functional (or the signed measure corresponding to it) is orthogonal to a subspace if it vanishes on all functions within that subspace.  
This abstract notion replaces the inner product-based orthogonality found in Hilbert spaces.



