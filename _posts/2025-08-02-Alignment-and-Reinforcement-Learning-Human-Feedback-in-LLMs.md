---
layout: post
title: "Alignment and Reinforcement Learning with Human Feedback in LLMs"
date: 2025-08-02 00:00:00 -0000
author: Santanu Pattanayak
tags: Reinforcement Learning, RL, Alignment in LLMs, RLHF  
---

# Table of Contents
1. [Introduction](#introduction)
2. [Proximal Policy Optimization ](#ppo)
3. [Training the Reward Model](#trm)
4. [Reward Model Construction from SFT Model](#Rrmcfsft)
5. [Direct Preference Optimization](#dpo)
6. [Derivation of DPO Objective from PPO and the Reward Model Training loss](#ppo2dpo)
7. [Feature Comparison between PPO style Alignment vs DPO](#ppovsdpo)
8. [Group Relative Policy Optimization](#grpo)
9. [Feature Comparison of PPO vs GRPO](#ppovsgrpo)


## Introduction <a name="introduction"></a>

Alignment represents a shift from traditional likelihood-based training. Rather than simply maximizing the probability of the next token, alignment focuses on steering a language model’s outputs toward human values, preferences, and intended goals. This approach is essential for mitigating issues such as harmful content, logical inconsistencies, and hallucinations—challenges that next-token prediction alone does not address or prevent. Alignment is mostly done with Reinforcement learning under the tag of Reinforcement Learning with Human Feedback (RLHF). 
The most commonly form of RLHF methods used are:  
1. Proximal Policy Optimization(PPO)
2. Direct Preference Optimization(DPO)
3. Group Relative Policy Optimization(GRPO)

## Proximal Policy Optimization <a name="ppo"></a>

Proximal Policy Optimization is a trust method Policy gradient method of Reinforcement Learning Technique where the updates to the policy in each step is done in such a way that the policy parameters doesn't change too much from that in the earlier iteration. Given a stochastic policy $$\pi$$ parameterized by $$\theta$$ that maps any state $$s$$ to an action $$a$$ probabilistically the update rule of PPO is given by  
In the context of the LLM alignment we assume that given a query $$x$$ the LLM which acts as a policy $$\pi_{\theta}$$ generates the output $$y$$ stochastically. We don't want to shift the RL aligned model parameters $$\theta$$ to be too far from the Supervised fine-tuned(SFT) model parameters  $$\theta_{SFT}$$. If we sample the queries $$x$$ from some dataset $$D_x$$ then the modified PPO objective for RLHF is as below 

$$
\begin{align}
L(\theta) &= \mathbb{E}_{x \sim D_x}\mathbb{E}_{y \sim \pi_{\theta}(y|x)}\left[r_{\phi}(x,y)\right] - \beta.KL(\pi_{\theta}(y|x) || \pi_{\theta_{SFT}}(y|x)) \\
&= \mathbb{E}_{x \sim D_x}\mathbb{E}_{y \sim \pi_{\theta}(y|x)}\left[r_{\phi}(x,y) - \beta \log\frac{\pi_{\theta}(y|x)}  {\pi_{\theta_{SFT}}(y|x)}\right] 
\end{align}
$$

The hyperparameter $$\beta$$ balances the reward maximization objective and the objective to prevent too much deviation of the model parameters from the SFT model capture through the KL divergence of the policies. 
The reward for completion $$y$$ given query $$x$$ which we have denoted by  $$r_{\phi}(x,y)$$ is generally computed from a trained reward model. Since the reward model assigns a score at the end of the completion of $$y$$ there isn't any reward after each token generation.

There are few differences between the PPO objective shown above modified for alignment from the traditional RL PPO objective. 

1. The reward from reward model is used directly and the formulation doesn't have any baseline subtraction from reward to work with Advantages.
2. The KL divergence in standard PPO is with respect to old policy $$\pi_{old}$$, which constraints the updates for exploration stability. In Alignment objective the KL divergence is with respect to the SFT model $$\pi_{SFT}$$ which ensures that the aligned model is not too different from the SFT model.

## Practical approach to PPO for Alignment 

As discussed in the previous section, the PPO objective used in InstructGPT [1] does not explicitly incorporate several common tweaks typically associated with PPO. It is unclear whether these modifications were applied during training in InstructGPT, but from a broader PPO implementation perspective it is worth revisiting these standard refinements.

* **Variance reduction with advantage estimation**:  To reduce the variance of the policy gradient estimate, it is common to work with the advantage function $$A(x,y)$$ instead of using the raw reward $$r_{\phi}(x,y)$$ directly. The advantage is obtained by subtracting a baseline value function that depends only on the prompt 
$$x$$. This baseline, denoted $$V_{\gamma}(x)$$ is typically estimated using a trained critic model. Formally, the advantage and the updated PPO objective are as follows:  

  $$
  \begin{align}
  &A_{\phi,\gamma}(x,y) = r_{\phi}(x,y) - V_{\gamma}(x) \\
  &L(\theta) = \mathbb{E}_{x \sim D_x}\mathbb{E}_{y \sim \pi_{\theta}(y|x)}\left[A_{\phi,\gamma}(x,y)\right] - \beta.KL(\pi_{\theta}(y|x) || \pi_{\theta_{SFT}}(y|x)) 
  \end{align}
  $$

* **Importance Sampling with Old policy** :  Importance sampling lets us rewrite expectations under the current policy $$\pi_{\theta}$$ in terms of expectations under the old policy $$\pi_{old}$$ . This allows us to use old trajectories or completions in this case while still estimating gradients for the current policy. This is a standard practice in PPO to enable sample efficiency. However, we should not reuse a very old policy for estimating a current policy. The importance sampling introduces a policy ratio $$\frac{\pi_{\theta}}{\pi_{old}}$$ because of the swapping of the expectation and the PPO objective can be modified as:  


  $$
  \begin{align}
  L(\theta) = \mathbb{E}_{x \sim D_x}\mathbb{E}_{y \sim \pi_{old}(y|x)}\left[\frac{\pi_{\theta}(y|x)}{\pi_{old}(y|x)} A_{\phi,\gamma}(x,y)\right] - \beta.KL(\pi_{\theta}(y|x) || \pi_{\theta_{SFT}}(y|x)) 
  \end{align}
  $$

* **Clipped objective for Stability**: PPO belongs to the family of **Trust Region Policy Optimization (TRPO)** methods, which aim to achieve approximately monotonic policy improvement across update iterations. To ensure stability, PPO prevents the updated policy $$\pi_{\theta}$$ from deviating too far from the old policy $$\pi_{old}$$ by clipping the policy ratio within a predefined range. The resulting clipped surrogate objective is given by:
  
  $$
  \begin{align}
  L^{CLIP}(\theta) &= \mathbb{E}_{x \sim D_x}\mathbb{E}_{y \sim \pi_{old}(y|x)}\left[\min(\frac{\pi_{\theta}(y|x)}{\pi_{old}(y|x)} A_{\phi,\gamma}(x,y),clip(1 - \epsilon,1 + \epsilon,\frac{\pi_{\theta}(y|x)}{\pi_{old}(y|x)}) A_{\phi,\gamma}(x,y) \right]  \\
  &- \beta.KL(\pi_{\theta}(y|x) || \pi_{\theta_{SFT}}(y|x)) 
  \end{align}
  $$



## Training the Reward Model <a name="trm"></a>

Instead of taking the feedback of the users to a completion based on a given query as the reward in alignment or RLHF, a reward model is trained. Here are the steps towards alignment as illustrated in the InstructGPT paper [1] 
1. First step is to sample prompts from a Prompts Dataset and a labeler comes up with the desired output behavior. This data is used for Supervised Finetuning.


2. Second step is where we build a reward model. For a given prompt, several model outputs are sampled. The labelers ranks the outputs from best to worst. This comparison data is used to train the reward model. Generally the SFT model from first step is taken as a starting point for the Reward Model.

3. Finally the SFT model from first step is optimized using the reward model using Reinforcement Learning.


<img width="760" height="493" alt="image" src="https://github.com/user-attachments/assets/db8438df-10a5-4b87-a3f7-6f3e5b11af73" />


Figure 1: Illustration of the SFT training, Reward Model training and Reinforcement learning for Alignment using Reward Model. 

## Reward Model Construction from SFT Model <a name="rmcfsft"></a>

1. The reward model architecture in used in RLHF pipelines such as InstructGPT or ChatGPT builds on top of the SFT Model architecture and its weights as the starting weights for the Reward Model with one exception - the final layer which gives the next token probability scores over the vocabulary is replaced by a layer that gives the reward the final output. 
2. The input to the reward model is the prompt $$x$$ along with the completion $$y$$ while the output is the reward $$r_{\phi}(x,y)$$
3. Training the Reward Model is on the preference data set $$D$$. Let's say for the prompt $$x$$ the completion $$y^{+}$$ is preferred over the completion $$y^{-}$$  
The reward model is trained with the softmax loss over the two completions $$y^{+}$$ and $$y^{-}$$ as follows:  

$$
\begin{align}
L(\phi) = -\mathbb{E}_{(x,y^{+},y^{-}) \sim D}  \log\left[\frac {\exp(r_{\phi}(x,y^{+}))} {\exp(r_{\phi}(x,y^{+})) + \exp(r_{\phi}(x,y^{-}))}\right] 
\end{align}
$$

The softmax is inspired by Bradley Terry Model which for preference pairs models the Probability of the preference completion as follows  :

$$
\begin{align}
\mathbb{P}(y^{+} > y^{-} | x ) = \frac {\exp(r_{\phi}(x,y^{+}))} {\exp(r_{\phi}(x,y^{+})) + \exp(r_{\phi}(x,y^{-}))} 
\end{align}
$$

One important aspect to note here, that we are not regressing on the reward $$r(x,y)$$ directly, but rather they act as logits for the completions $$y$$ given the prompt $$x$$. 

## Direct Preference Optimization <a name="dpo"></a>

Direct Preference Optimization(DPO) [2] is a RL technique for Alignment which skips training a reward model and subsequently performing RL. Instead, given preference pairs sampled from a preference dataset $$ x,y^{+}, y^{-} \sim D$$ updates the Language model directly instead of first building a reward model with the preference dataset and then optimizing through RL using the same. The same is illustrated in the image below taken from the DPO paper.

<img width="926" height="190" alt="image" src="https://github.com/user-attachments/assets/823dd379-7cc9-457a-94bb-e7a0ca222f32" />


Figure 2: DPO optimizing for human preferences while avoiding reinforcement learning.

If the LLM we want to align through DPO is presented by parameterized policy $$\pi_{\theta}(.)$$ while the SFT LLM is represented as  $$\pi_{SFT}(.)$$ given the preference dataset $$D$$ which contains tuples of $$x,y^{+},y^{-}$$ where $$x$$ is the prompt $$y^{+},y^{-}$$ are the winning and losing completions given the prompt, the DPO loss is given as follows:  

$$
\begin{align}
L(\pi_{\theta},\pi_{SFT}) = -\mathbb{E}_{x,y^{+},y^{-} \sim D}  \log\left[ \frac{1}{1 + \exp{(\beta \log\frac{\pi_{\theta}(y^{-}|x)}{\pi_{SFT}(y^{-}|x)}} - \beta \log\frac{\pi_{\theta}(y^{+}|x)}{\pi_{SFT}(y^{+}|x)})}  \right]
\end{align}
$$




## Derivation of DPO Objective from PPO and the Reward Model Training loss <a name="ppo2dpo"></a>

Let's look at the PPO objective and try to get an optimal policy $$\pi^{*}$$ by minimizing the same.

$$
\begin{align}
L(\theta) &= \mathbb{E}_{x \sim D_x}\mathbb{E}_{y \sim \pi_{\theta}(y|x)}\left[r(x,y) - \beta \log\frac{\pi_{\theta}(y|x)}  {\pi_{\theta_{SFT}}(y|x)}\right] 
\end{align}
$$

Diving both sides by $$\beta$$ just scales the DPO objective and hence the optimal policy derived by minimizing the same wont change. Also, to bring the beta normalized reward to log scale we exponentiate it so that it can be tied to the log associated with the policies. See below:  


$$
\begin{align}
L(\theta) &= \mathbb{E}_{x \sim D_x}\mathbb{E}_{y \sim \pi_{\theta}(y|x)}\left[\frac{1}{\beta}r(x,y) -  \log\frac{\pi_{\theta}(y|x)}  {\pi_{SFT}(y|x)}\right]  \\
&= \mathbb{E}_{x \sim D_x}\mathbb{E}_{y \sim \pi_{\theta}(y|x)}\left[\log \exp(\frac{r(x,y)}{\beta})-  \log\frac{\pi_{\theta}(y|x)}  {\pi_{SFT}(y|x)}\right]  \\
&= \mathbb{E}_{x \sim D_x}\mathbb{E}_{y \sim \pi_{\theta}(y|x)}\left[-  \log\frac{\pi_{\theta}(y|x)}  {\exp(\frac{r(x,y)}{\beta}) \pi_{SFT}(y|x)}\right] 
\end{align}
$$

If we normalize
$$\exp(\frac{r(x,y)}{\beta})\pi_{SFT}(y|x)$$  
by the partition function $$Z(x)$$ to sum over all $$y$$ given $$x$$ we would be sure the same would be a probability and hence a policy which we denote by $$\pi^{*}$$. See below 

$$
\begin{align}
Z(x) = \sum_{y}{\exp(\frac{r(x,y)}{\beta}) \pi_{SFT}(y|x)} \\
\pi^{*}(y|x) = \frac{1}{Z(x)}{\exp(\frac{r(x,y)}{\beta}) \pi_{SFT}(y|x)}
\end{align}
$$

Using the partition function $$Z(x)$$ and subsequently the policy $$\pi^{*}$$ we can modify the PPO loss as below:  


$$
\begin{align}
L(\theta) &= \mathbb{E}_{x \sim D_x}\mathbb{E}_{y \sim \pi_{\theta}(y|x)}\left[-  \log\frac{\pi_{\theta}(y|x)}  {\exp(\frac{r(x,y)}{\beta}) \pi_{SFT}(y|x)}\right] \\
&=\mathbb{E}_{x \sim D_x}\mathbb{E}_{y \sim \pi_{\theta}(y|x)}\left[-  \log\frac{\pi_{\theta}(y|x)}  {\frac{1}{Z(x)}\exp(\frac{r(x,y)}{\beta}) \pi_{SFT}(y|x)}+ \log Z(x)\right] \\
&=\mathbb{E}_{x \sim D_x}\mathbb{E}_{y \sim \pi_{\theta}(y|x)}\left[-  \log\frac{\pi_{\theta}(y|x)}  {\pi^{*}(y|x)}+ \log Z(x)\right] \\
&=\mathbb{E}_{x \sim D_x}-\mathbb{E}_{y \sim \pi_{\theta}(y|x)}  \log\frac{\pi_{\theta}(y|x)}  {\pi^{*}(y|x)}+ \mathbb{E}_{x \sim D_x}\log Z(x) \\
&=\mathbb{E}_{x \sim D_x} - KL(\pi_{\theta}||\pi^{*}) + c
\end{align}
$$

We can see from above to maximize the PPO objective we need to minimize the KL divergence and KL divergence is minimum when the distribution match. Hence, our desired optimal policy $$\pi_{\theta}$$ should be as follows:  

$$
\begin{align}
\pi_{\theta}(y|x) = \pi^{*}(y|x) = \frac{1}{Z(x)}{\exp(\frac{r(x,y)}{\beta}) \pi_{SFT}(y|x)}
\end{align}
$$

If we were to express the reward as a function of the policies from the above we would get 

$$
\begin{align}
r(x,y) =  \beta \log \frac{\pi_{\theta}(y|x)}{\pi_{SFT}(y|x)} + \beta Z(x)
\end{align}
$$


Now substituting the deduced $$r(x,y)$$ into the Bradley Terry Preference Model we get:  


$$
\begin{align}
\mathbb{P}(y^{+} > y^{-} | x ) &= \frac {\exp(r(x,y^{+}))} {\exp(r(x,y^{+}) + \exp(r(x,y^{-}))} \\
&= \frac {1} {1 + \exp(r(x,y^{-}) - r(x,y^{+})) } \\
&= \frac {1} {1 + \exp(\beta \log \frac{\pi_{\theta}(y^{-}|x)}{\pi_{SFT}(y^{-}|x)} + \beta Z(x)) - \beta \log \frac{\pi_{\theta}(y^{+}|x)}{\pi_{SFT}(y^{+}|x)} - \beta Z(x))) } \\
&= \frac {1} {1 + \exp(\beta \log \frac{\pi_{\theta}(y^{-}|x)}{\pi_{SFT}(y^{-}|x)}  - \beta \log \frac{\pi_{\theta}(y^{+}|x)}{\pi_{SFT}(y^{+}|x)}) } \\
\end{align}
$$

Taking negative logarithm on either side of the above equation we get the objective for DPO alignment as we can see below:

$$
\begin{align}
-\log \mathbb{P}(y^{+} > y^{-} | x )&= -\log\left[ \frac {1} {1 + \exp(\beta \log \frac{\pi_{\theta}(y^{-}|x)}{\pi_{SFT}(y^{-}|x)}  - \beta \log \frac{\pi_{\theta}(y^{+}|x)}{\pi_{SFT}(y^{+}|x)}) }\right] 
\end{align}
$$  


## Feature Comparison between PPO style Alignment vs DPO <a name="ppovsdpo"></a>


| **Feature / Aspect**          | **InstructGPT (PPO)**                         | **DPO**                                               |
|------------------------------|-----------------------------------------------|-------------------------------------------------------|
| **Reward model required**     |  Yes                                          |  No                                                   |
| **Alignment signal**          | Scalar reward (from reward model)             | Pairwise preferences directly                         |
| **Policy optimization**       | PPO (actor-critic, clipped surrogate loss)    | Supervised-style loss from preference modeling        |
| **KL regularization**         |  Explicit KL penalty                          |  Implicit via preference loss structure               |
| **Training stability**        |  Sensitive to hyperparameters                 |  Very stable                                          |
| **Implementation complexity** | High – reward model + PPO loop + rollout buffer | Low – single pass with preference-labeled data        |
| **Exploration**               |  Possible via sampling during PPO rollouts    |  Not included – works only with offline preference data |
| **Sample efficiency**         |  Low – requires rollouts + reward model scoring |  High – no reward model or environment interaction    |
| **Risk of reward hacking**    |  Moderate – depends on reward model quality   |  Low – no scalar reward to over-optimize              |
| **Compute cost**              | High – due to reward model inference + RL rollouts | Low – simple gradient-based updates                   |
| **Scalability**               |  Harder to scale due to PPO complexity        |  Highly scalable – behaves like supervised fine-tuning |


## Group Relative Policy Optimization <a name="grpo"></a>


**Group Relative Policy Optimization (GRPO)** is an alternative to PPO in which the baseline is not estimated by a critic model $$V_{\gamma}$$. Instead, the baseline is computed directly from the rewards of multiple sampled completions $$\{y_1,y_2,...y_n\}$$ having rewards  $${r_2,...r_n\}$$ to form baseline.

In general the advantage $$A_i$$ for a query $$x$$ with completion $$y_i$$ and reward signal $$r_i$$ is as below  

$$
\begin{align}
A_i = \frac {r_i - \bar{r}}{\sigma}
\end{align}
$$

where  :

$$
\begin{align}
\bar{r} &= \frac{1}{n} \sum_{j=1:n} r_j  \\
\sigma^{2}  &= \frac{1}{n-1} \sum_{j=1:n} (r_j - \bar{r})^{2}
\end{align}
$$


<img width="1000" height="400" alt="image" src="https://github.com/user-attachments/assets/f7be0b02-186e-4748-ba5c-9065f084945f" />





Figure 3: GRPO illustration

The optimization framework for GRPO is much like PPO except for the baseline value computation without a trained critic, as illustrated in Figure-3.
Hence, the optimization objective is also similar to PPO as shown below:  

  $$
  \begin{align}
 L(\theta) &= \mathbb{E}_{x \sim D_x}\frac{1}{n}\sum_{i=1:n}\left[\min(\frac{\pi_{\theta}(y_i|x)}{\pi_{old}(y_i|x)} A_{\phi}(x,y_i),clip(1 - \epsilon,1 + \epsilon,\frac{\pi_{\theta}(y_i|x)}{\pi_{old}(y_i|x)}) A_{\phi}(x,y_i) \right]  \\
  &- \beta.KL(\pi_{\theta}(.|x) || \pi_{\theta_{SFT}}(.|x)) 
  \end{align}
  $$

Although the objective looks the same few subtle differences of this GRPO objective from PPO are:  
* The advantage doesn't involve the critic model $$V_{\gamma}$$ and hence the advantage function only depends on reward model parameters. Hence while the advantage in PPO is parameterized by reward model parameter $$\phi$$ and critic model parameter $$\gamma$$ in GRPO the advantage is solely a function reward model.
* GRPO uses the Monte Carlo approximation of KL divergence which ensure per sample the KL divergence penalty is positive.

## Feature Comparison of PPO vs GRPO <a name="ppovsgrpo"></a>

| Aspect | **PPO**                                                                  | **GRPO**                                                     |
|--------|--------------------------------------------------------------------------|--------------------------------------------------------------|
| **Baseline** | Uses critic network $$V_{\gamma}(x)$$                                    | Uses group mean reward $$\bar{r}$$                           |
| **Critic requirement** | Needs training of a separate critic (can be unstable)                    | No critic needed → simpler pipeline                          |
| **Sample efficiency** | More sample-efficient (critic generalizes across prompts)                | Less efficient (requires multiple completions per prompt)    |
| **Computation cost** | Lower (one completion and one critic evaluation)                         | Higher (multiple completions per prompt to compute baseline) |
| **Variance in advantage** | Depends on critic accuracy; poor critic → high variance                  | Variance reduced by averaging group rewards                  |
| **Adaptability to non-stationary rewards** | Critic may lag if reward distribution shifts                             | Baseline recomputed per batch → more adaptive                |
| **Theoretical grounding** | Stronger links to trust-region methods, monotonic improvement guarantees | More heuristic, weaker formal guarantees                     |
| **Implementation** | More complex (policy, critic and  reward model)                            | Simpler (policy and reward model only)                       |


## References

[1] [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)  

[2] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)

[3] [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300)







































