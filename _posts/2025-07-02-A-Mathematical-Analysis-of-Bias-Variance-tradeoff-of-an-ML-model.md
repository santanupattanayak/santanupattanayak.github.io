---
layout: post
title: "A Mathematical Analysis of Bias-Variance tradeoff of an ML model"
date: 2025-07-02 00:00:00 -0000
author: Santanu Pattanayak
tags: Bias, Variance. 
---

# Table of Contents
1. [Introduction](#introduction)
2. [Bias Variance and Irreducible Error decomposition](#bvd)

## Introduction <a name="introduction"></a>

Whenever we build a model, it is not feasible to train on all datapoints of feature vector $$x$$ and their corresponding target $$y$$ under the distribution $$P(x,y)$$.  Instead, we draw a training dataset $$D$$ of $$m$$ points from $$P(x,y)$$:  

$$
\begin{align}
D={(x_1,y_1), (x_2,y_2), .... (x_m,y_m)}
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



<img width="1300" height="450" alt="image" src="https://github.com/user-attachments/assets/c075678b-7496-4b64-a0a6-2d9c20e2b854" />




Figure-2.  Error for a given test input x  


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

Let us look at the **second term** which we are able to factorize as product of expectations as the model predictions $$\hat{y{_D}}$$ doesn't have noise distribution dependency which the noise $$\epsilon$$ doesnt have data distribution dependency :

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


Finally lets target the **1st term** which would give us our **bias** and **variance** component of the test loss. For starters we will add and subract the mean of the predictions over various model $$\mathop {\mathbb E}_ {D \sim P(D)} [\hat{y{_D}}] $$ as it concerns both the bias and variance. To avoid clutter of notation we will just refer to it as $${\mathbb E} [\hat{y{_D}}] $$ in the below deduction.  

$$
\begin{align}

&\mathop {\mathbb E}_ {D \sim P(D)} \mathop {\mathbb E}_{\epsilon \sim N(0,\sigma^{2})} [(\bar{y}   - \hat{y{_D}} )^{2}] \\
&= \mathop {\mathbb E}_ {D \sim P(D)} \mathop {\mathbb E}_{\epsilon \sim N(0,\sigma^{2})} [(\bar{y}   - {\mathbb E} [\hat{y{_D}}] + {\mathbb E} [\hat{y{_D}}] -\hat{y{_D}} )^{2}] \\
&= \mathop {\mathbb E}_ {D \sim P(D)} \mathop {\mathbb E}_{\epsilon \sim N(0,\sigma^{2})} [(\bar{y}   - {\mathbb E} [\hat{y{_D}}])^{2} + ({\mathbb E} [\hat{y{_D}}] -\hat{y{_D}})^{2} + 2(\bar{y}   - {\mathbb E} [\hat{y{_D}}])({\mathbb E} [\hat{y{_D}}] -\hat{y{_D}})]
\end{align}
$$
 
Let us inspect the final product term here. The noise distribution doesn't involve either of the product term and hence can be eliminated. 

$$
\begin{align}
\mathop {\mathbb E}_ {D \sim P(D)} \mathop {\mathbb E}_{\epsilon \sim N(0,\sigma^{2})} 2(\bar{y}   - {\mathbb E} [\hat{y{_D}}])({\mathbb E} [\hat{y{_D}}] -\hat{y{_D}}) \\
= \mathop {\mathbb E}_ {D \sim P(D)} 2(\bar{y}   - {\mathbb E} [\hat{y{_D}}])({\mathbb E} [\hat{y{_D}}] -\hat{y{_D}})
\end{align}
$$


The first product term $$(\bar{y}   - {\mathbb E} [\hat{y{_D}}])$$  is constant with respect to the distribution of training dataset $$D$$ and hence can be taken out of expectation. We denote it by $$C$$. So the product term simplifies to 

 $$C{\mathop {\mathbb E}_ {D \sim P(D)} ({\mathbb E} [\hat{y{_D}}] -\hat{y{_D}})} $$


Since $${\mathop {\mathbb E}_ {D \sim P(D)} ({\mathbb E} [\hat{y{_D}}] -\hat{y{_D}})}= 0$$ the product term vanishes from the **1st term** and we are left with 

$$ \mathop {\mathbb E}_ {D \sim P(D)} \mathop {\mathbb E}_{\epsilon \sim N(0,\sigma^{2})} [(\bar{y}   - {\mathbb E} [\hat{y{_D}}])^{2} + ({\mathbb E} [\hat{y{_D}}] -\hat{y{_D}})^{2} $$

None of the remaining terms is dependent on the noise distribution and hence we can rewrite the terms as 

$$ \mathop {\mathbb E}_ {D \sim P(D)}  [(\bar{y}   - {\mathbb E} [\hat{y{_D}}])^{2} + (\hat{y{_D}} - {\mathbb E}[\hat{y{_D}}] )^{2}] $$

So combining the components left after all the **simplifications of the 3 terms** in the test loss $$L$$:

$$ L = \mathop {\mathbb E}_ {D \sim P(D)}  [(\bar{y}   - {\mathbb E} [\hat{y{_D}}])^{2} + (\hat{y{_D}} - {\mathbb E}[\hat{y{_D}}] )^{2}]  + \sigma^{2} $$

The first term and second term are nothing but the square of the **bias** and the **variance** of the model respectively as we have defined earlier. The final term is the **irreducible noise variance**. So we can see that the test loss can be decomposed into the bias and variance of the model along with the irreducible noise component. 
