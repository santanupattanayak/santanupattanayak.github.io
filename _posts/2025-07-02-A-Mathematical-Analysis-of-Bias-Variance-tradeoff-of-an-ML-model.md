---
layout: post
title: "Mathematics of Bias-Variance Decomposition and Emerging Tradeoffs in Overparameterized Neural Networks"
date: 2025-07-02 00:00:00 -0000
author: Santanu Pattanayak
tags: Bias, Variance. 
---

# Table of Contents
1. [Introduction](#introduction)
2. [Bias Variance and Irreducible Error decomposition](#bvd)
3. [Bias Variance Tradeoff](#bvt)
4. [Rethinking Bias-Variance Trade-off for Over parameterized Networks](#rtbvt)
5. [Understanding Double Descent using Linear Regression](#tradeoff)
6. [Conclusion](#conclusion)

## Introduction <a name="introduction"></a>

Whenever we build a model, it is not feasible to train on all datapoints of feature vector $$x$$ and their corresponding target $$y$$ under the distribution $$P(x,y)$$.  Instead, we draw a training dataset $$D$$ of $$m$$ points from $$P(x,y)$$:  

$$
\begin{align}
D=\{(x_1,y_1), (x_2,y_2), .... (x_m,y_m)\}
\end{align}
$$

These $$m$$ datapoints are independently  and identically sampled from  $$P(x,y)$$.  

Since $$m$$ data-points from the distribution $$P(x,y)$$ can be chosen in multiple ways the training dataset $$D$$ itself follows a distribution:  

$$
\begin{align}
D \sim P^{m} (x,y)
\end{align}
$$

as illustrated in Figure-1.  

For a given model class and training methodology each dataset $$D$$ results in a different model parameterized by $$\theta_{D}$$, which leads to variability in the model prediction $$\hat{y_{D}}$$.  This variability known as **variance of the model** is quantified as: 

$$
\begin{align}
\mathop {\mathbb E}_{D \sim P(D)} ({\theta{_D}} - {\mathbb E}    [{\theta{_D}}])^{2} 
\end{align}
$$      



Equivalently, in terms of variance over the predictions given an input $$x$$ over the different models trained with datasets $$D \sim P(D)$$  :  


$$
\begin{align}
\mathop {\mathbb E}_{D \sim P(D)} (\hat{y{_D}} - {\mathbb E}    [{\hat y{_D}}])^{2} 
\end{align}
$$

We will use this prediction-based form for variance in our analysis.



<img width="1300" height="500" alt="image" src="https://github.com/user-attachments/assets/5a166d40-4ee1-422c-a186-0625bd663249" />


Figure-1. Illustration of Variance of a Model  


Another source of unpredictability is the inherent noise in the target $$y$$ -  it is not fully predictable from $$x$$ for most of the applications.  
For regression problems, which we would use to illustrate this bias variance tradeoff (See Figure-2), the target is generally modeled as  

$$
\begin{align}
y = \bar{y}(x) + \epsilon ,  \epsilon \sim N(0,\sigma^{2})
\end{align}
$$


In essence $$y$$ given $$x$$ follows a normal distribution as below:  

$$
\begin{align}
y|x \sim N(\bar{y}(x),\sigma^{2})
\end{align}
$$

and hence best prediction we can make is just the mean of the distribution 

$$
\begin{align}
{\mathbb E}(y|x) = \bar{y}(x)
\end{align}
$$

The residue $$\epsilon$$ represents an **irreducible error**(see Figure 2.0)  that the model cannot predict.  

A good model pertaining to any dataset $$D$$ should produce predictions $$y_D$$ close to $$\bar{y}(x)$$ for an input $$x$$. 
If the expectation of the predictions $$\hat {y_D}$$ over the model distribution(resulting from distribution over $$D$$) equals $$\bar{y}$$ then the model would be an unbiased estimator of predictable component $$\bar{y}$$.  
Hence the **bias of the model** is defined as  

$$
\begin{align}
\mathop {\mathbb E}_{D \sim P(D)} \bar{y} - [\hat{y{_D}}] 
\end{align}
$$


So bias is the model's inability to catch up to the predictable component of the target. It often stems from the model being too simplistic to capture the nuances in the input to output relation. For instance using a Linear regression model to solve a problem where the dependency between input and target is non linear might introduce high bias in the model.



<img width="1300" height="500" alt="image" src="https://github.com/user-attachments/assets/c075678b-7496-4b64-a0a6-2d9c20e2b854" />




Figure-2.  Error for a given test input $$x$$  


## Bias Variance and Irreducible Error decomposition <a name="bvd"></a>

We will choose to look at the test loss for a given input vector $$x$$. As I have illustrated in Figure-2 given the input $$x$$ the variability in the target $$y$$ is because of the distribution over $$y$$ given $$x$$ i.e $$P(y|x) ~ N(\bar{y_{x}}, \sigma^2) $$ . 
The mean of the conditional distribution  (we will drop the suffix $$x$$ from $$\bar{y}_{x} $$ for ease of notation)  is the predictable component of the $$y$$ and the variance is because of the unpredictable noise component $$\epsilon$$ and as discussed earlier  $$y = \bar{y} + \epsilon$$

Similarly, the variability in the model prediction $$\hat{y_{D}}$$ is because of the distribution over the training datasets leading to different different model parameters $$\theta{_D}$$ pertaining to each dataset $$D \sim P(D)$$


$$
\begin{align}
L = \mathop {\mathbb E}_ {D \sim P(D)} \mathop {\mathbb E}_{y \sim N(\bar{y},\sigma^{2})} (y - \hat{y{_D}} )^{2} 
\end{align}
$$

We can replace $$y$$ by $$\bar{y} + \epsilon$$ in $$L$$ and change the expectation over $$y \sim N(\bar{y},\sigma^{2})$$ to expectation over the noise $$\epsilon \sim N(0,\sigma^{2})$$. 

$$
\begin{align}
L &= \mathop {\mathbb E}_ {D \sim P(D)} \mathop {\mathbb E}_{\epsilon \sim N(0,\sigma^{2})} (\bar{y} + \epsilon - \hat{y{_D}} )^{2} \\
&= \mathop {\mathbb E}_ {D \sim P(D)} \mathop {\mathbb E}_{\epsilon \sim N(0,\sigma^{2})} [(\bar{y}   - \hat{y{_D}} )^{2} + 2\epsilon (\bar{y}   - \hat{y{_D}} ) + \epsilon^{2}] 
\end{align}
$$

Let us look at the **second term** which we are able to factorize as product of expectations as the model predictions $$\hat{y{_D}}$$ doesn't have noise distribution dependency while the noise $$\epsilon$$ doesn't have data distribution dependency :

$$
\begin{align}
\mathop {\mathbb E}_ {D \sim P(D)} \mathop {\mathbb E}_{\epsilon \sim N(0,\sigma^{2})}  [2\epsilon (\bar{y}   - \hat{y{_D}} )]  \\

= 2 \mathop {\mathbb E}_ {D \sim P(D)}[(\bar{y} - \hat{y_{D}})]        \mathop {\mathbb E}_{\epsilon \sim N(0,\sigma^{2})}  [\epsilon]
\end{align}
$$

Since  the mean of the noise $$\mathop {\mathbb E}_{\epsilon \sim N(0,\sigma^{2})}  [\epsilon] = 0 $$ hence the second term is 0. 
 
The **third term** also simplifies to be the noise variance which is the irreducible component of the error.

 $$
\begin{align}
\mathop {\mathbb E}_ {D \sim P(D)} \mathop {\mathbb E}_{\epsilon \sim N(0,\sigma^{2})}  [\epsilon^{2}] \\
 = \mathop {\mathbb E}_ {D \sim P(D)} [\sigma^{2}]
\end{align}
$$
 
 Since $$\sigma$$ doesn't depend on the distribution over $$D$$ hence 
$$
\begin{align}
\mathop {\mathbb E}_ {D \sim P(D)} [\sigma^{2}] = \sigma^{2}
\end{align}
$$


Finally lets target the **1st term** which would give us our **bias** and **variance** component of the test loss. For starters we will add and subtract the mean of the predictions over various model $$\mathop {\mathbb E}_ {D \sim P(D)} [\hat{y{_D}}] $$ as it concerns both the bias and variance. To avoid clutter of notation we will just refer to it as $${\mathbb E} [\hat{y{_D}}] $$ in the below deduction.  

$$
\begin{align}

&\mathop {\mathbb E}_ {D \sim P(D)} \mathop {\mathbb E}_{\epsilon \sim N(0,\sigma^{2})} [(\bar{y}   - \hat{y{_D}} )^{2}] \\
&= \mathop {\mathbb E}_ {D \sim P(D)} \mathop {\mathbb E}_{\epsilon \sim N(0,\sigma^{2})} [(\bar{y}   - {\mathbb E} [\hat{y{_D}}] + {\mathbb E} [\hat{y{_D}}] -\hat{y{_D}} )^{2}] \\
&= \mathop {\mathbb E}_ {D \sim P(D)} \mathop {\mathbb E}_{\epsilon \sim N(0,\sigma^{2})} [(\bar{y}   - {\mathbb E} [\hat{y{_D}}])^{2} + ({\mathbb E} [\hat{y{_D}}] -\hat{y{_D}})^{2} \\
&+ 2(\bar{y}   - {\mathbb E} [\hat{y{_D}}])({\mathbb E} [\hat{y{_D}}] -\hat{y{_D}})]
\end{align}
$$
 
Let us inspect the final product term here. The noise distribution doesn't involve either of the product term and hence can be eliminated. 

$$
\begin{align}
\mathop {\mathbb E}_ {D \sim P(D)} \mathop {\mathbb E}_{\epsilon \sim N(0,\sigma^{2})} 2(\bar{y}   - {\mathbb E} [\hat{y{_D}}])({\mathbb E} [\hat{y{_D}}] -\hat{y{_D}}) \\
= \mathop {\mathbb E}_ {D \sim P(D)} 2(\bar{y}   - {\mathbb E} [\hat{y{_D}}])({\mathbb E} [\hat{y{_D}}] -\hat{y{_D}})
\end{align}
$$


The first product term $$(\bar{y}   - {\mathbb E} [\hat{y{_D}}])$$  is constant(say $$c$$) with respect to the training dataset distribution $$P(D)$$and hence the product term simplifies to:   


$$
\begin{align}
{\mathop {\mathbb E}_ {D \sim P(D)} c({\mathbb E} [\hat{y{_D}}] -\hat{y{_D}})}
\end{align}
$$



Since $${\mathop {\mathbb E}_ {D \sim P(D)} ({\mathbb E} [\hat{y{_D}}] -\hat{y{_D}})}= 0$$ the product term vanishes from the **1st term** and we are left with 

$$ \mathop {\mathbb E}_ {D \sim P(D)} \mathop {\mathbb E}_{\epsilon \sim N(0,\sigma^{2})} [(\bar{y}   - {\mathbb E} [\hat{y{_D}}])^{2} + ({\mathbb E} [\hat{y{_D}}] -\hat{y{_D}})^{2} $$

None of the remaining terms is dependent on the noise distribution and hence we can rewrite the terms as 

$$ \mathop {\mathbb E}_ {D \sim P(D)}  [(\bar{y}   - {\mathbb E} [\hat{y{_D}}])^{2} + (\hat{y{_D}} - {\mathbb E}[\hat{y{_D}}] )^{2}] $$

So combining the components left after all the **simplifications of the 3 terms** in the test loss $$L$$:

$$ L = \mathop {\mathbb E}_ {D \sim P(D)}  [(\bar{y}   - {\mathbb E} [\hat{y{_D}}])^{2} + (\hat{y{_D}} - {\mathbb E}[\hat{y{_D}}] )^{2}]  + \sigma^{2} $$

The first term and second term are nothing but the square of the **bias** and the **variance** of the model respectively as we have defined earlier. The final term is the **irreducible noise variance**. So we can see that the test loss can be decomposed into the bias and variance of the model along with the irreducible noise component. 

## Traditional Bias Variance Tradeoff <a name="rbvt"></a> 

A model can suffer from **high bias** when its architecture is overly simplistic—for example, choosing a shallow network instead of a deeper one. Increasing the model’s complexity generally helps reduce this bias by allowing it to capture more of the underlying patterns in the data.  

However, **pushing complexity too far**, especially when the **available dataset is small**, often leads to **overfitting**. In that case, the model starts fitting not just the underlying signal but also the noise present in the training data.
When overfitting occurs, the training loss may look excellent, but the model **fails to generalize to unseen data**, even if that data carries a similar underlying signal. Here, by signal we mean the predictable component of the target, as opposed to the random fluctuations or noise.

In traditional ML models, as **model complexity increases**, the **bias typically decreases monotonically** because the model can capture more of the **underlying structure in the data**. At the same time, the **variance increases monotonically** since the model becomes more **sensitive to fluctuations or noise** in the training set.
The generalization error (test loss) initially decreases with complexity, since the drop in bias dominates. Beyond a certain point, however, the rising variance outweighs the bias reduction, causing the generalization error to increase again. This gives rise to the well-known **U-shaped curve** of the bias–variance tradeoff, as shown below in Figure 3.

<img width="800" height="450" alt="image" src="https://github.com/user-attachments/assets/e52db487-4f11-4997-87b0-82bf63dcaa12" />

Figure-3 Traditional Bias Variance Tradeoff Curve


## Rethinking Bias-Variance Trade-off for Over parameterized Networks <a name="rtbvt"></a>  

The traditional bias-variance tradeoff does not hold in the case of over-parameterized neural networks. The same is the theme of the paper [1] While bias still decreases monotonically with increasing model complexity, variance does not follow the expected upward trend. Instead, variance typically rises initially as the width of the network (i.e., the number of neurons per layer) increases, but then begins to decline with further increases in width—resulting in a uni-modal variance curve.  

This **non-monotonic behavior of variance** gives rise to the **double descent pattern** in generalization error. As model complexity increases, the generalization error first decreases, then spikes near the interpolation threshold(where model parameters equals training samples and hence training error can be zero), and finally descends again as the model becomes increasingly over-parameterized. See Figure-4 illustrated below. 

<img width="941" height="300" alt="image" src="https://github.com/user-attachments/assets/2ee2fa25-6b34-44e0-9f96-03f2330f68a2" />

Figure-4.  Double Descent generalization pattern  

With increase in model complexity the bias as we can see from the plots above goes down monotonically, while the variance has a uni-modal trend with its peak near the interpolation zone, where the overfitting is highest. 
Near the interpolation zone, where the model parameters equal the number of datapoints, the variance is maximum as the model overfits to the data. 


## Understanding Double Descent using Linear Regression <a name="tradeoff"></a>  

Let us build some intuition for why the **double descent generalization pattern** appears in the over-parameterized regime of neural networks by first examining a simple linear regression model.  

Suppose we have $$m$$ data points and $$n$$ parameters. When $$m > n$$ (i.e., more data points than parameters), the system of equations is overdetermined and admits no exact solution in general. In this case, given a data matrix $$X \in \mathbb{R}^{m \times n}$$ and a target vector $$Y \in \mathbb{R}^m$$, the least-squares solution is  

$$
\theta = (X^{\top}X)^{-1}X^{\top}Y.
$$  

It is important to note that the target $$y$$ for feature vector $$x$$ can be decomposed into a predictable component $$\bar{y}(x)$$ and an unpredictable noise term $$\epsilon$$:  

$$
y(x) = \bar{y}(x) + \epsilon.
$$  

Because the system is overdetermined ($$m > n$$), the least-squares solution cannot achieve zero error in general. In particular, it cannot fit the entire noise component of $$y$$. However, as the number of parameters $$n$$ increases, the model gains more flexibility and can begin fitting not only the true signal but also portions of the noise. This increasing capacity to overfit noise is precisely what drives the rise in variance and generalization error as we approach the interpolation threshold—before eventually decreasing again in the highly over-parameterized regime.  

Now let us examine what happens at the **interpolation threshold**, where the number of parameters equals the number of data points ($$n = m$$). In this case, the system of linear equations admits an exact solution:  

$$
\theta = X^{-1}Y.
$$  

At this point, the model is forced to interpolate the training data exactly, fitting both the true signal and the noise without distinction. As a result, the variance of the model is maximized, which explains the peak in generalization error observed at the interpolation threshold. 

Let us try to analyse the variance upto the interpolation zone mathematically using properties from Random matrix theory.
Given a data matrix \(X\), the covariance of the linear regression parameter estimates is  

$$
\operatorname{Cov}(\theta \mid X) = \sigma^{2}(X^{\top}X)^{-1}
$$

If the rows of \(X\) are sampled i.i.d. from a multivariate normal distribution with covariance \(\Sigma\), then the scatter matrix  

$$
S = X^{\top}X \;\sim\; \mathcal{W}_n(\Sigma, m),
$$ 

follows a Wishart distribution with $$m$$ degrees of freedom and scale $$\Sigma$$. A standard result for the Wishart is  

$$
\mathbb{E}[S^{-1}] = \frac{1}{m-n-1}\,\Sigma^{-1}, \quad m>n+1
$$

Using this result, the expected covariance of the regression parameters becomes  

$$
\begin{align}
\mathbb{E}_{X}\!\left[\operatorname{Cov}(\theta \mid X)\right] &= \sigma^{2}\,\mathbb{E}_{S}[S^{-1}] 
&= \sigma^{2}\,\frac{1}{m-n-1}\,\Sigma^{-1}, \quad m>n+1
\end{align}
$$

As the number of parameters $$n$$ increases relative to the number of samples $$m$$, the variance of the estimator grows, and near the interpolation threshold $$m \approx n+1$$, the variance diverges. This explains why the variance of linear regression estimators peaks sharply around the **interpolation threshold**.


Beyond the **interpolation threshold**, when the number of parameters $$n$$ exceeds the number of data points $$m$$, the system becomes under-determined and admits infinitely many solutions. Optimization methods such as gradient descent tend to favor *minimum-norm* (or low-norm) solutions. In this case, the solution can be written as  

$$
\theta = X^{\top}(XX^{\top})^{-1}y.
$$  

For the underdetermined system  

$$
X\theta = Y, \quad \text{with } \operatorname{rank}(X) = m,
$$  

the general solution can be decomposed into two parts: a component in the **row space** of $$X$$ and a component in the **null space** of $$X$$. That is,  

$$
\theta = \theta_{\text{row}} + \theta_{\text{null}}, 
\quad \text{where } X\theta_{\text{null}} = 0.
$$  

Any row-space solution can be expressed as a linear combination of the rows of $$X$$:  

$$
\theta_{\text{row}} = X^{\top}k.
$$  

Substituting this into the system yields  

$$
X\theta_{\text{row}} = Y 
\;\;\implies\;\;
XX^{\top}k = Y 
\;\;\implies\;\;
k = (XX^{\top})^{-1}Y.
$$  

Thus, the row-space solution is  

$$
\theta_{\text{row}} = X^{\top}(XX^{\top})^{-1}Y.
$$  

Since this solution lies entirely in the row space of $$X$$ and excludes any null-space component, it is the *minimum-norm solution*. Gradient descent, when initialized at zero, naturally converges to this solution in the over-parameterized regime.  

Crucially, low-norm solutions have a reduced tendency to overfit noise. Hence, beyond the interpolation threshold, the variance of the model decreases because the additional degrees of freedom allow the optimizer to select such low-norm solutions.  

This behavior, illustrated in the linear regression case, extends to neural networks as well and gives rise to the characteristic **double descent** pattern in generalization.  



## Conclusion <a name="conclusion"></a>  

The bias–variance framework provides a powerful lens to understand the generalization behavior of machine learning models. In classical settings, model complexity trades off bias for variance, leading to the well-known U-shaped test error curve. However, in modern over-parameterized neural networks, this picture is incomplete.  

Through the lens of linear regression, we saw why the **double descent** phenomenon emerges:  
- In the **overdetermined regime** ($$m > n$$), models cannot perfectly fit noise, and variance gradually increases as capacity grows.  
- At the **interpolation threshold** ($$m = n$$), the model fits both signal and noise exactly, maximizing variance and producing the worst generalization.  
- In the **overparameterized regime** ($$n > m$$), optimization biases (e.g., gradient descent favoring minimum-norm solutions) reduce variance again, leading to improved generalization despite massive model capacity.  

This shift—from the traditional U-shaped curve to the double descent curve—captures a core insight into why highly overparameterized models such as deep neural networks often generalize surprisingly well in practice.  

Ultimately, double descent illustrates that **more parameters do not always imply worse generalization**. Instead, implicit biases in optimization and the geometry of high-dimensional solution spaces play a critical role. Understanding these principles is not only theoretically important but also practically relevant as we continue to build increasingly large and expressive models.  



## References 
1. [Rethinking Bias-Variance Trade-off for Generalization of Neural Networks](https://arxiv.org/pdf/2002.11328)
