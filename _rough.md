### Linear Functional in Finite Dimension
For a **finite-dimensional normed linear space** such as $$ \mathbb{R}^n $$, which is a Hilbert space, linear functionals act on vectors to produce scalars in $$ \mathbb{R} $$.  
It can be shown that any linear functional $$ L $$ acts on each vector $$ x \in \mathbb{R}^n $$ as:

$$
L(x) = a^{T}x,
$$

where the vector $$ a $$ corresponds to an element of the **dual space** $$ X^* $$. This is called the **Reisz representation theorem**.

Since in an $$L^2$$ space, the elements of the dual coincide with those of the original space; hence $$ a \in X $$ as well.

**Key points:**
* Any functional $$ L $$ in a finite-dimensional normed vector space corresponds to some element $$ a $$ in the dual space $$ X^* $$, which itself forms a normed linear space.

* For an $$ L^2 $$ (Hilbert) space, $$ X^* = X $$; thus, linear functionals correspond to vectors in $$ X $$ acting on other vectors via the **inner product** (dot product).

* If $$ U $$ is a closed convex subset of $$ X $$, it is orthogonal to the space $$ X - U $$. Hence any element $$ a \in X - U $$ is orthogonal to all elements in $$ U $$, i.e.  
   $$ a^{T}x = 0 \quad \forall x \in U. $$  
   Since $$ a $$ corresponds to a functional, any $$ L $$ in the dual space of $$ X - U $$ satisfies  
   $$ L(x) = 0 \quad \forall x \in U. $$

### Linear Functional in Infinite Dimension Space
For an **infinite-dimensional normed linear space** such as $$ C[a,b] $$, which consists of all continuous functions on the closed and conpact interval $$[a,b] $$, the form of linear functionals differs from those in finite-dimensional spaces.  
Since $$C[a,b]$$ is equipped with the **supremum norm** (also called the $$ L^{\infty} $$ norm), it is **not self-dual** unlike Hilbert spaces with the $$ L^2 $$ norm.  
As a result, the linear functionals on $$ C[a,b] $$ are generally not elements of $$C[a,b]$$ itself, but instead belong to its dual space, which has a richer and more abstract structure.

Example of **Functional** on $$C[a,b]$$

Let $$X = C[a,b]$$ with the sup norm $$\|f\|_\infty = \sup_{x \in [a,b]} |f(x)|$$ . 
We can define a linear functional $$L_{a}$$ on $$X$$ such that  

$$ 
L_{a}(f) = \int_a^b f(x)\,dx 
$$  

The functional is bounded since 
$$
\|L_{a}(f)\| \le (b - a)\|f\|_\infty 
$$

Another example of a functional on $$X$$ is the **evaluation functional**, which maps each function $$f \in X$$ to its value at a fixed point $$x_0 \in [a,b]$$:

$$
L_{b}(f) = f(x_0), \quad \forall f \in X.
$$

Another slightly modified version of the evaluation functional given two fixed points $$x_{0},x_{1} \in [a,b]$$ can be  

$$
L_{c}(f) = f(x_0) - f(x_1), \quad \forall f \in X.
$$

In the case of a **finite-dimensional $$L^2$$ normed linear space** $$X$$, every linear functional corresponds to an element of the same space $$X$$, since the dual space $$X^*$$ is isomorphic to $$X$$ itself. Each such functional acts on the elements of $$X$$ via the **dot product**, as established by the **Riesz Representation Theorem**.

For **infinite-dimensional normed spaces**, such as the **$$L^{\infty}$$ space**, the Riesz Representation Theorem takes a different form compared to $$L^2$$ (Hilbert) spaces. Unlike $$L^2$$ spaces, $$L^{\infty}$$ is **not self-dual**. Its **dual space** $$\left(L^{\infty}\right)^*$$ is much larger and is given by the space of **bounded finitely additive signed measures** on the underlying measurable space.

Thus, for every continuous linear functional $$L$$ on $$L^{\infty}$$, there exists a finitely additive signed measure $$\nu$$ such that:

$$
L(f) = \int_X f(x) \, d\nu(x), \quad \forall f \in L^{\infty}
$$

Measures in mathematics are a way of assigning positive value (can be some notion of length, area, volume) to certain subsets of a given set. A signed measure also allows for negative weight. We will not go into technical depth of measures but rather look at how these signed measures look for the functionals $$L_a,L_b,L_c$$.  
The signed measures for the functional $$L_{a}$$ is $$d\nu = dx$$.
The measure for $$L_{b}$$ is $$d\nu = \delta(x - x_{0})dx$$ as it collapses the function at $$x_{0}$$ and for similar reasons the measure for $$L_{c}$$ is $$d\nu = (\delta(x - x_{0}) - \delta(x - x_{1}))dx$$ 

**Key Points**
* The collection of all **bounded linear functionals** on $$X$$ forms a normed linear space called the **dual space**, denoted $$X^*$$.

* In $$L2$$ space the notion of orthogonality was easy as it has a geometric intuition, any vector $$a$$ corresponding to a functional $$L$$ is orthogonal to $$x$$ if $$L(x) = a^{T}x = 0 $$. Simply the dot product of the element $$a$$ corresponding to the functional $$L$$ and vector $$x$$ is zero when $$a \perp x$$ . 
This allowed us to see any element $$a \in (X-U)^* =(X-U)$$ are all orthogonal to all elements in closed and convex subset $$U \subset X$$
In $$L^{\infty}$$ space the notion of orthogonality is given by linear functionals in its dual such that if for a Linear functional $$L \in X^*$$ given by some signed measure gives $$L(x) = 0$$, then we say the signed measure is orthogonal to $$x$$.
   With this notion of orthogonality, if $$U$$ is a closed subspace of infinite dimensional $$L^{\infty}$$ space $$X$$ then there would exist functionals L in $$X^*$$ which would annihilate all of $$U$$ as follows:  
   $$
   L(x) = 0 \forall x \in U
   $$
   The orthogonality in $$L^{\infty}$$ is not in the geometric sense but in the sense of function vanishing on the subspace.
   In this sense the orthogonal complement $$U^{\perp}$$ can be defined as 

   $$
   U^{\perp} = \{ L |  L(x) = 0 \forall x \in U\}
   $$


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
