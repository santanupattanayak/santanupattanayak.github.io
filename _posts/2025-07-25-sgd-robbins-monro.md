# Stochastic Gradient Descent Through the Lens of the Robbins-Monro Algorithm

## Introduction to Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) is a variant of Gradient Descent where, instead of computing the gradient over the entire dataset in each iteration, we compute it using a randomly sampled datapoint. When each iteration uses a subset of datapoints \( m \ll N \), where \( N \) is the total number of datapoints, it is known as **Minibatch Gradient Descent**.

If the loss for a datapoint \( (x, y) \) is \( \ell(x, y) \), the total expected loss is:

$$
L(\theta) = \mathbb{E}_{x,y \sim P(x,y)} [\ell(\theta, x, y)]
$$

The gradient of the loss becomes:

$$
\nabla_{\theta} L(\theta) = \mathbb{E}_{x,y \sim P(x,y)} \left[ \nabla_{\theta} \, \ell(\theta, x, y) \right]
$$

In SGD, this expectation is approximated using a single datapoint's gradient, which is an unbiased estimator of the full gradient:

$$
\mathbb{E}_{x,y \sim P(x,y)} \left[ \nabla_{\theta} \, \ell(\theta, x, y) \right] = \nabla_{\theta} L(\theta)
$$

For minibatches of \( m \) i.i.d. samples \( B = \{(x_i, y_i)\}_{i=1}^m \), the gradient estimate is:

$$
\mathbb{E}_{x,y \sim P(x,y)} \left[\frac{1}{m} \sum_{i=1}^m \nabla_{\theta} \, \ell(\theta, x_i, y_i) \right]
= \nabla_{\theta} L(\theta)
$$

## Unbiasedness in the Finite Dataset Setting

Assume we have a dataset of size \( N \). The full dataset gradient is:

$$
\nabla_{\theta} L(\theta) = \frac{1}{N} \sum_{i=1}^N \nabla_{\theta} \, \ell(\theta, x_i, y_i)
$$

The expected gradient from a random minibatch of size \( m \) is:

$$
\mathbb{E}_B \left[\frac{1}{m} \sum_{i=1}^m \nabla_{\theta} \, \ell(\theta, x_i, y_i) \right] = \nabla_{\theta} L(\theta)
$$

Thus, SGD uses an unbiased estimate of the true gradient. The update rule is:

$$
\theta^{(t+1)} = \theta^{(t)} - \eta_{t} \nabla_{\theta} L_{m}(\theta)
$$

where \( t \) is the iteration index, \( \eta_t \) is the learning rate, and \( L_m(\theta) \) is the minibatch loss.

## Robbins-Monro Algorithm

The Robbins-Monro algorithm solves root-finding problems of the form \( g(x) = 0 \), even when \( g(x) \) is not directly observable. If we can only observe noisy values:

$$
\tilde{g}(x) = g(x) + \epsilon
$$

and

$$
\mathbb{E}_{\epsilon}[\tilde{g}(x)] = g(x)
$$

then we can apply the update rule:

$$
x_{t+1} = x_t - \eta_t \tilde{g}(x_t)
$$

Convergence requires:

$$
\sum_{t=1}^{\infty} \eta_t = \infty, \quad \sum_{t=1}^{\infty} \eta_t^2 < \infty
$$

## SGD as Robbins-Monro

Minimizing

$$
L(\theta) = \mathbb{E}_{x,y \sim P(x,y)}[\ell(x, y)]
$$

is equivalent to solving

$$
\nabla_{\theta} L(\theta^*) = 0
$$

Since

$$
\nabla_{\theta} L(\theta) = \mathbb{E}_{x,y}[\nabla_{\theta} \ell(\theta, x, y)]
$$

we cannot compute this exactly in large-scale settings. Instead, we use the minibatch gradient \( \nabla_{\theta} L_m(\theta) \), which satisfies:

$$
\mathbb{E}[\nabla_{\theta} L_m(\theta)] = \nabla_{\theta} L(\theta)
$$

Hence, SGD can be interpreted as an instance of the Robbins-Monro algorithm:

$$
\theta^{(t+1)} = \theta^{(t)} - \eta_t \nabla_{\theta} L_m(\theta)
$$

Thus, **SGD is a practical application of Robbins-Monro stochastic approximation** to modern optimization problems.
