---
layout: post
title: "Contraction Mapping and Convergences in Machine Learning"
date: 2025-07-02 00:00:00 -0000
author: Santanu Pattanayak
tags: banach fixed point, Contraction Mapping, gradient descent convergence through Contraction Mapping. 
---

## Contraction Mapping 

1. A mapping or transform $$T$$ on a complete metric space $$(X,d)$$ onto itself is said to be a Contraction mapping if $$T$$ brings the points closer to each other. Mathematically, $$T:X \leftarrow X$$ if said to be a contraction mapping if there exists $$0 \le c \le 1$$ such that

$$ \lVert Tx - Ty \rVert \le c \lVert x-y \rVert \: \forall x,y \in X $$


2. When $$c \< 1 $$ we call it a strict contraction. A strict contraction has a unique fixed point $$x$$ that satisfies $$Tx = x$$

3. In general the **Contraction Mapping theorem** states that if $$T:X \rightarrow X$$ is a contraction mapping in a **complete metric space** $$(X,d)$$ then there is exactly one fixed point $$x \in X $$ that satisfies $$Tx=x$$  


