---
layout: post
title: "Stochastic Gradient Descent Through the Lens of Robbins Monro Algorithm"
date: 2025-07-14 00:00:00 -0000
author: Santanu Pattanayak
tags: banach fixed point, Contraction Mapping, gradient descent convergence through Contraction Mapping. 
---

## Introduction

Stochastic Gradient Descent is a variant of Gradient Descent where instead of computing the Gradient over an entire dataset at all iterations, we compute the gradient over a small set of randomly sampled datapoints.
When the randomly sampled datapoints in each iteration is $$1$$ it's called Stochastic Gradient Descent when the number of points is some $$m << N$$ where $$N$$ is the number of training datapoints its called a Minibatch Gradient Descent.

