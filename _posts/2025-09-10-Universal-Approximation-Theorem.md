---
layout: post
title: "Universal Approximation theorem"
date: 2025-08-09 00:00:00 -0000
author: Santanu Pattanayak
tags: UAT, Universal approximation theorem, Neural networks, 
---

# Table of Contents
1. [Introduction to Universal Approximation theorem ](#introduction)


## Introduction

The Universal Approximation Theorem (UAT) is one of the cornerstones of modern deep learning theory. At its heart, it asserts that a sufficiently large neural network can approximate any continuous function to arbitrary accuracy under certain conditions. While the statement is popular in machine learning circles, its mathematical foundations are deeply rooted in **real analysis** and **functional analysis**.
This post explores the UAT, its formal statement, and the real analysis theorems that underlie it.

## Norm of a function
The norm of a function would appear multiple times in our discussion of UAT and hence it makes sense to go over it before we dive into the various aspects of UAT.
The Euclidean or the $$l_2$$ norm of a vector $$x$$ in a d-dimensional vector space is defined as  :

$$
\begin{align}
\|x\|_{2} =   \sum_{i=1}^{d} |x|^{2} 
\end{align}
$$

The $$l_p$$ norm generalizes the $$l_2$$ to any value $$p$$ and is similarly defined as  :
$$
\begin{align}
\|x\|_{p} =   \sum_{i=1}^{d} |x|^{p} 
\end{align}
$$

$$L^{p}$$ norm is extension of the $$l_p$$ norm of vectors to functions over defined domain $$D$$. Any function $$f(x)$$ can be viewed as a vector of values over the different values of input $$x$$ over the domain. The $$L^{p}$$ norm of a function $$f(x) , x \in D$$ is given by:  
$$
\begin{align}
\|f\|_{p} = \int_{x \sim D} |f(x)|^{p} dx
\end{align}
$$


## The Universal Approximation Theorem

The earliest rigorous versions were proved independently by Cybenko (1989) and Hornik, Stinchcombe, and White (1989). One simplified version is:

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



Supremum norm is defined as the max absolute difference between the function over the domain $$[0,1]^{n} $$ of the function.


Supremum norm is actually $$L^{p}$$ norm where $$p=\infty$$. 

$$
\begin{align}
\|f\|_{p} = \int_{x \sim D} |f(x)|^{p} dx
\end{align}
$$

When $$p=2$$ we get $$L^{2}$$ norm for which is the extension of the $$l_2$$ norm or Euclidean norm defined over vectors. We can think of functions as vectors with finite or infinite dimensions corresponding to the various inputs in its domain.   




