---
layout: post
title: "How Neural Networks Approximate Any Function: A Mathematical Dive into Universal Approximation theorem"
date: 2025-08-09 00:00:00 -0000
author: Santanu Pattanayak
tags: UAT, Universal approximation theorem, Neural networks, 
---

# Table of Contents
1. [Introduction to Universal Approximation theorem](#introduction)
2. [Norm of a function](#norm)
3. [The Early Versions of Universal Approximation Theorem](#uat)
4. [Practical Illustration why Universal Approximation Theorem works](#practical)
5. [Mathematical Concepts Review to understand Universal Approximation theorem](#maths)
6. [Cybenko’s Universal Approximation Theorem — A Mathematically Intuitive Proof](#proof)
7. [Conclusion](#conclusion)

## Introduction <a name="introduction"></a>

The Universal Approximation Theorem (UAT) is one of the cornerstones of modern deep learning theory. At its heart, it asserts that a sufficiently large neural network can approximate any continuous function to arbitrary accuracy under certain conditions. While the statement is popular in machine learning circles, its mathematical foundations are deeply rooted in **normed linear spaces**, **real analysis**, **functional analysis** as well as in "measure theory"
This post explores the UAT, its formal statement, and the real analysis theorems that underlie it.

## Norm of a function <a name="norm"></a>
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

The earliest rigorous versions of UAT were proved independently by Cybenko (1989)[1] and Hornik, Stinchcombe, and White (1989). One simplified version is:

Let $$\sigma: \mathbb{R} \rightarrow \mathbb{R} $$ be any continuous squashing function such as sigmoid. Then for any continuous function $$f$$ defined on the cube $$I_n = [0,1]^{n}$$ and for any tolerance $$\epsilon \gt 0$$ there exists a neural network of the form

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



## Practical Illustration why Universal Approximation Theorem works <a name="practical"></a>

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





## Mathematical Concepts Review to understand Universal Approximation theorem <a name="maths"></a>

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
Its dual space $$C[a,b]^*$$ consists not of continuous functions, but of more general objects—specifically, **bounded finitely additive signed measures** on $$[a,b]$$.  

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
L(f) = \int_a^b f(x)\ d\mu(x).
$$

#### High level brief of measures 
In classical calculus, integration is done with respect to a variable $$x$$(written as $$dx$$). In measure theory, we integrate a function with respect to a measure $$\mu$$(written as $$d\mu$$) which generalizes the idea of length, area, volume or probability.
Integration of a function $$f$$ with respect to a measure $$\mu$$ over a set $$A$$ is written as

$$
L(f) = \int_{x \in A} f(x)\, d\mu(x).
$$ 

The measure is defined over "interesting" subsets of $$A$$ and while integrating set a disjoint set of subsets $$\{A_i\}_i$$ of $$A$$ the union of which should be $$A$$.
These subsets $$A_i$$ can be singleton sets consisting of single element.  Do note that these "interesting subsets" of $$A$$ forms a collection called **sigma algebra** $$\mathcal{F}$$ and the measure is a function defined on the subsets of sigma algebra and not on the outcomes $$x \in A$$. To keep things contained we will not discuss measurable spaces in details hence , I advise readers unfamiliar with the same to go through it to get a good sense of measure.  

The integration wrt to the measure denotes sum the values of $$f(x)$$ weighted by the measure $$\mu$$ of the set $$A$$ around each point $$x$$.
Basically measure generalizes the concept of integration over sets.

**Probability measure** - Let $$X$$ be random variable mapping the elements $$\omega \in \Omega$$ to $$\mathbb{R}$$ i.e. $$X:\Omega \rightarrow \mathbb{R}$$
  Then the expectation of the random variable is expressed in terms of measure as below 

$$
\int_{\omega} X(\omega)\ d\mu(w)
$$
If the probability density of the random variable which is our called measure here is $$p(x)$$ we have $$d\mu(x) = p(x)dx$$ and hence we can write the integral as 

$$
\int_{x \in R} X d\mu(x) = \int_{x \in R} Xp(x)dx
$$

**Evaluation measure**
The functional $$L_{x_0}(f) = f(x_0)$$ corresponds to the **Dirac measure** $$\mu = \delta_{x_0}$$, so  
  $$
  L_{x_0}(f) = \int_a^b f(x)\, d\delta_{x_0}(x)
  $$

**Signed measure**
When measures can take negative value we called it a signed measure. The difference functional $$L(f) = f(x_1) - f(x_2)$$ corresponds to the **signed measure** $$\mu = \delta_{x_1} - \delta_{x_2}$$
As stated earlier the dual of the $$L^{\infty}$$ space consists of signed measures.

### Orthogonality and Duality in Infinite Dimensions

In $$\mathbb{R}^n$$, orthogonality has a clear **geometric interpretation**: two vectors are orthogonal if their inner product is zero.  

In infinite-dimensional spaces such as $$C[a,b]$$, this geometric notion no longer applies directly. Instead, orthogonality is defined in terms of the **annihilation** of functionals on subspaces.  

If $$M$$ is a closed subspace of $$C[a,b]$$, the **orthogonal complement** $$M^\perp \subset C[a,b]^*$$ consists of all functionals $$L$$ in the dual space such that  

$$
L(f) = 0 \quad \text{for all } f \in M.
$$

In this sense, the orthogonality in $$C[a,b]$$ is not geometric but **functional**: a functional (or the signed measure corresponding to it) is orthogonal to a subspace if it vanishes on all functions within that subspace.  
This abstract notion replaces the inner product-based orthogonality found in Hilbert spaces.

This notion of orthogonality in $$L^{\infty}$$ space is also called Hahn Banach Separation principal.


## Cybenko’s Universal Approximation Theorem — A Mathematically Intuitive Proof <a name="proof"></a>

Now that we have defined some of the foundational mathematical ideas, let us walk through **Cybenko’s** original proof of the **Universal Approximation Theorem**.



### **Premise**

Consider the space of continuous functions $$C(I_n), \quad \text{where } I_n = [0,1]^n $$ the **n-dimensional unit cube**, which serves as our domain.

We define the family of functions of the form $$ g(x) = \sum_{i=1}^{m} \alpha_i \, \sigma(w_i \cdot x + b_i) $$  

where  
- $$\sigma$$ is a fixed **activation function**,  
- $$w_i \in \mathbb{R}^n$$, $$b_i \in \mathbb{R}$$ are  hidden layer parameters 
- $$\alpha_i \in \mathbb{R}$$ are output layer linear coefficients combining the nonlinear sigmoid activations.

Let this collection of functions formed by finite linear combinations of the sigmoid activations in the hidden layer be denoted by $$S \subset C(I_n)$$.

### **Density Claim**

Cybenko’s theorem states that the set $$S$$ is **dense** in $$C(I_n)$$ with respect to the **supremum norm**.  
That is, for every continuous function $$f \in C(I_n)$$ and every $$ \varepsilon > 0 $$, there exists a function $$g \in S $$ such that

$$
\| f - g \|_{\infty} < \varepsilon
$$

Equivalently, the **closure** of $$S$$, denoted $$ \overline{S}$$, equals the entire space $$C(I_n)$$.



### **Proof by Contradiction**

Suppose, for contradiction, that $$ \overline{S} \neq C(I_n)$$.  
Then the closure of $$S$$, call it $$R = \overline{S}$$, is a **proper closed subspace** of $$C(I_n)$$.

### **Introducing Orthogonality**

By the **Hahn–Banach separation principle**, there exists a **non-zero bounded linear functional**  
$$L \in C(I_n)^*$$ (the dual space) such that

$$
L(f) = 0 \quad \forall f \in R.
$$

Since $$S \subset R$$, it follows that

$$
L(f) = 0 \quad \forall f \in S.
$$

### **Riesz Representation**

From the **Riesz Representation Theorem**, every bounded linear functional on $$C(I_n)$$ can be represented as an **integral against a finite signed measure** $$\mu$$ on $$I_n$$:

$$
L(f) = \int_{I_n} f(x) d\mu(x)
$$
for some $$\mu \neq 0$$ in the dual space $$M(I_n) = C(I_n)^*$$



### **Applying Reiz representation and Orthogonality to the Sigmoid Function Family**

Since each $$g \in S$$ has the form $$ g(x) = \sum_{i=1}^{m} \alpha_i \, \sigma(w_i \cdot x + b_i) $$ the orthogonality condition $$L(g) = 0$$ gives  

$$
\int_{I_n} \sigma(w \cdot x + b)\, d\mu(x) = 0 
\quad \text{for all } w \in \mathbb{R}^n, \, b \in \mathbb{R}
$$

### **Role of Discriminatory Functions**

A function $$\sigma$$ is said to be **discriminatory** if the only measure $$\mu$$ satisfying

$$
\int_{I_n} \sigma(w \cdot x + b)\, d\mu(x) = 0
\quad \text{for all } w,b
$$
is the **zero measure**, i.e., $$\mu = 0$$.

Common activation functions like the **sigmoid** satisfy this property.

Since our assumption led to the existence of a **non-zero** signed measure $$\mu$$ that makes all these integrals vanish, this contradicts the discriminatory property of $$\sigma$$.



### **Conclusion** <a name="conclusion"></a>

Hence, our assumption that $$\overline{S} \neq C(I_n)$$ must be false.  
Therefore, $$S$$ is **dense** in $$C(I_n)$$, proving that finite linear combinations of the form

$$
\sum_{i=1}^{m} \alpha_i \, \sigma(w_i \cdot x + b_i)
$$
can approximate any continuous function on $$I_n$$ arbitrarily well.


##Conclusion
The Universal Approximation Theorem serves as a cornerstone connecting mathematical theory with the practical capabilities of neural networks. It assures us that, under mild conditions on the activation function, even a single-hidden-layer network can approximate any continuous function on a compact domain to arbitrary precision.  

Cybenko’s proof reveals the deeper analytical structure behind this fact — drawing upon concepts like linear functionals, signed measures, and the Hahn–Banach separation theorem. The key lies in the discriminatory property of the activation function: if no nontrivial signed measure can annihilate all activations, the span of these activations must be dense in the space of continuous functions.  

However, the theorem is existential, not constructive. It tells us such networks exist but says nothing about the number of neurons required or how to efficiently find their parameters through training. In practice, extremely wide single-layer networks may be needed to approximate highly complex functions.  

This is where depth becomes crucial. Adding more hidden layers allows neural networks to represent compositional structures, enabling them to approximate complex, hierarchical functions with far fewer neurons. Deeper networks can capture intricate features and dependencies that would require an impractically wide shallow network to represent.  

In essence, the Universal Approximation Theorem guarantees that neural networks are capable of representing any function — while the theory of deep architectures explains how they can do so efficiently. Representational power is not the bottleneck; the real challenges lie in optimization, generalization, and interpretability. The theorem thus remains a guiding principle, illuminating the mathematical foundation beneath the empirical success of deep learning.


## References

[1] Approximation by Superpositions of a Sigmoidal Function : https://web.njit.edu/~usman/courses/cs675_fall18/10.1.1.441.7873.pdf  



