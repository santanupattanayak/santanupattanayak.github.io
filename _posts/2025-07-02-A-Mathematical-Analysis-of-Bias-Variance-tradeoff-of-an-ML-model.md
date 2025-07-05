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
$$D \sim P^{n} (x_i,y_i) $$.

3. Given a model class and training methodology each dataset $$D$$ would produce a different model $$f(y|x;\theta_{D})$$ and that's what lead to the variance of the model. Because of this variability for a given $x$ models trained on different datasets $$D$$ would produce different predictions $$\hat{y_{D}} = f(y|x;\theta_{D})$$ . The variance of the model can be represented in terms of variance over the parameters of the models trained with datasets $$D \sim P(D)$$

$$\mathop {\mathbb E}_{D \sim P(D)} ({\theta{_D}} - {\mathbb E}    [{\theta{_D}}])^{2} $$
   
Similarly it can be expressed as variance over the predictions over the different models trained with datasets $$D \sim P(D)$$ given an input $$x$$ i.e.   

$$\mathop {\mathbb E}_{D \sim P(D)} ({y{_D}} - {\mathbb E}    [{y{_D}}])^{2} $$

We will choose this prediction representation for variance of model for our  bias-variance decomposition analysis.

5. The other source of unpredictibility as well as variability in prediction comes from the fact that the target $$y$$ is not fully predictable from $$x$$ for most of the applications. For example for regression problems which we would use to illustrate this bias variance tradeoff, the target is generally modeled as  $$y = \bar{y}(x) + \epsilon$$  where $$\epsilon \sim N(0,\sigma^{2})$$. In essence $$y$$ given $$x$$ follows a normal distribution  $$y|x \sim N(\bar{y}(x),\sigma^{2})$$ and hence best prediction we can make is just the mean of the distribution i.e. $$E(y|x) = \bar{y}(x)$$
So even if theoretically we are able to train on all data from the distribution with the right model, the error in prediction during test time cannot be zero as well will see. 

With the given information let's try to do a bias variance decomposition of the test error.

