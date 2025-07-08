---
layout: post
title: "A Mathematical Analysis of Bias-Variance tradeoff of an ML model"
date: 2025-07-02 00:00:00 -0000
tags: Bias, Variance. 
---

## Introduction

1. Whenever we build a model, it is not feasible to train on all datapoints of feature vector $$x$$ and their corresponding target $$y$$ under the distribution $$P(x,y)$$.  We sample a set of $$m$$ points from $$P(x,y)$$ which we call as training dataset $$D$$. The training dataset $$D$$ of $$m$$ points can be represented as $$D={(x_1,y_1), (x_2,y_2), .... (x_m,y_m)}$$
where each of the datapoints $$(x_i,y_i)$$ are independently  and identically sampled from  $$P(x,y)$$.

2. Since $$m$$ data-points from the distribution $$P(x,y)$$ can be chosen in multiple ways the training dataset $$D$$ has a distribution which follows   
$$D \sim P^{m} (x_i,y_i) $$ as illustrated in Figure-1.

3. Given a model class and training methodology each dataset $$D$$ would produce a different model parameterized by $$\theta_{D}$$ and that's what leads to the **variance of the model** . Because of this variability for a given input vecgtor $$x$$ models trained on different datasets $$D$$ would produce different predictions $$\hat{y_{D}}$$ . The variance of the model can be represented in terms of variance over the parameters of the models trained with datasets $$D \sim P(D)$$

$$\mathop {\mathbb E}_{D \sim P(D)} ({\theta{_D}} - {\mathbb E}    [{\theta{_D}}])^{2} $$      



Similarly it can be expressed as variance over the predictions over the different models trained with datasets $$D \sim P(D)$$ given an input $$x$$ i.e.   

$$\mathop {\mathbb E}_{D \sim P(D)} (\hat{y{_D}} - {\mathbb E}    [{\hat y{_D}}])^{2} $$

We will choose this prediction representation for variance of model for our  bias-variance decomposition analysis.



<img width="968" alt="image" src="https://github.com/user-attachments/assets/5a166d40-4ee1-422c-a186-0625bd663249" />


    Figure-1. Illustration of Variance of a Model




4. The other source of unpredictibility as well as variability in prediction comes from the fact that the target $$y$$ is not fully predictable from $$x$$ for most of the applications. For example for regression problems which we would use to illustrate this bias variance tradeoff, the target is generally modeled as  $$y = \bar{y}(x) + \epsilon$$  where $$\epsilon \sim N(0,\sigma^{2})$$ (See Figure-2). In essence $$y$$ given $$x$$ follows a normal distribution  $$y|x \sim N(\bar{y}(x),\sigma^{2})$$ and hence best prediction we can make is just the mean of the distribution i.e. $$E(y|x) = \bar{y}(x)$$. This leads to an **irreducible error** $$\epsilon$$ that the model can't predict. If the chosen model class and the training methodology is good, for a feature vector $$x$$ the predictions $$y_D$$ pertaining to the models for each dataset  $$D \sim P^m(x,y)$$ should be as close as possible to predictable component of $$y$$ that is $$\bar{y}$$. Infact the model predictions $$\hat {y_D}$$ would be an unbiased estimator of predictable component $$\bar{y}$$ if
$$\mathop {\mathbb E}_{D \sim P(D)} \hat{y{_D}} = \bar{y}$$ and hence the **bias of the model** is defined as 

$$\mathop {\mathbb E}_{D \sim P(D)} \bar{y} - [\hat{y{_D}}] $$

So bias is the model's inability to catch up to the predictable component of the target. It often stems from the model being too simplistic to capture the nuanaces in the input to output relation. For instance using a Linear regression model to solve a problem where the dependency between input and target is non linear might introduce high bias in the model.



<img width="1049" alt="image" src="https://github.com/user-attachments/assets/c075678b-7496-4b64-a0a6-2d9c20e2b854" />




    Figure-2.  Error for a given test input x  


## Bias Viariance and Irreducible Error decomposition

We will chose to look at the test loss for a given input vector $$x$$. As I have illustrated in Figure-2 given the input $$x$$ the variability in the target $$y$$ is because of the distribution over $$y$$ given $$x$$ i.e $$P(y|x) ~ N(\bar{y_{x}}, sigma^2) $$ . 
The meaan of the conditional distribution is the predictable component So(we will drop the suffix $$x$$ from $$\bar{y}_{x} $$ for ease of notation)  is the predictable component of the $$y$$  


