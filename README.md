# Reinforcement Learning Baselines

This is my collection of reinforcement learning baseline applied on a wide variety of OpenAI gym environments.

## Types of environments
There are two types of environments a RL agent can work in:
* Fully observable environment
* Partially observable environments


## Components of RL Agent
* Policy : agents behavious function
* Value function : how good is each state or action
* Model : agent's representation of the environment (predicts the next state and reward)


## Categorizing RL agents
* Brute force
    * Sample every policy
    * Choose best return
* Value based
    * No policy
    * Value function
* Policy based
    * Policy
    * No value function
* Actor Critic
    * Policy
    * Value function
* Model Free
    * Policy or Value function
    * No model
* Model based
    * Policy or Value function
    * Model

## Further class mterial from David Silver's RL course
* Model-free prediction
    * Monte-Carlo learning
    * Temporal-Difference learning
    * TD-lambda
* Model-free control
    * On-policy Monte Carlo Control
    * On-policy Temporal difference learning
    * Off-policy learning
* Value function approximation
    * Incremental methods
    * Batch methods
* Policy gradient
    * Finite difference Policy gradient
    * Monte-Carlo policy gradient
    * Actor-Critic policy gradient

## Easy21 submission
* Implementation of Easy21
* Monte-Carlo Control in Easy21
* TD Learning in Easy21
* Linear Function Approximation in Easy21