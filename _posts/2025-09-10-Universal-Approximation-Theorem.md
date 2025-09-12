---
layout: post
title: "Universal Approximation theorem"
date: 2025-08-09 00:00:00 -0000
author: Santanu Pattanayak
tags: UAT, Universal approximation theorem, Neural networks, 
---

# Table of Contents
1. [Introduction to Universal Approximation theorem ](#introduction)


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

## What guarantees  Universal Approximation Theorem <a name="uat"></a>

The big question is how do we know neural nets can approximate any function?  
This is where real analysis theorems come in. Given any domain  say $$D = [0,1]^{n}$$ we need to find a family of functions which can approximate any continuous function over the domain $$D$$. Generally such a **function space** is denoted by $$C([0,1]^{n})$$.  

When we talk about spaces of real $$\mathbb{R}$$ and rational numbers $$\mathbb{Q}$$ we say $$\mathbbQ$$ is **dense** in $$\mathbb{R}$$ since any real number is arbitrarily close to some rational number. In essence if we draw an open set of any non-zero radius $$r$$ around any real number $$x$$ denoted by $$(x-r,x+r)$$ it is bound to contain one or more rational numbers. 
And hence computers which store everything in rational numbers can represent any real number with negligible error. 


