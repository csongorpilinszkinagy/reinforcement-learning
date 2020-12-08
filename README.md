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


## Easy21 documentation

### RL cheat sheet
* Reward
    * Variable name: R_t
    * Scalar feedback signal
    * Indicates how well the agent is doing at step t
    * Agents job is to maximize the cumulative reward
    * Reward Hypothesis: every goal can by described by the maximisation of expected cumulative rewards
* Agent
    * Variable name: agent
    * Executes action A_t
    * Reveives observation O_t
    * Receives scalar reward R_t
* Environment
    * Variabel name: env
    * Receives action A_t
    * Emits observation O_t+1
    * Emits scalar reward R_t+1
* History
    * Variable name: H_t
    * Sequence of Obsevations, Rewards and Actions
    * H_t = O_1, R_1, A_1, ... , A_t-1, O_t, R_t
* State
    * Variable name: S_t
    * The information used to determine what happens next
    * Function of the history
    * S_t = f(H_t)
    * A Markov state contains all useful information from the history
    * "The future is independent of the past given the present"
* Policy
    * Variable name: policy
    * The agent's behaviour, a map from the state space to the environment space
    * Can be deterministic or stochastic
    * Deterministic policy:
        * action = policy(state)
    * Stochastic policy:
        * policy(action|state) = P[A_t = a|S_t = s]
* Value function
    ...

