import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from easy21 import Environment

from monte_carlo_agent_evaluation import MonteCarloAgentEvaluation
from monte_carlo_agent_control import MonteCarloAgentControl

def plot_agent_value_function(name, agent):
    fig = plt.figure('Agent value function', figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')

    value_function = agent.get_value_function()
    
    # TODO: get these from the environment
    X, Y = np.meshgrid(np.arange(1, 22, 1), np.arange(1, 11, 1))

    ax.set_xlabel('Player Sum')
    ax.set_ylabel('Dealer showing')
    ax.set_zlabel('Value')
    # TODO: get these from the environment
    ax.set_xticks(np.arange(1, 22, step=5))
    ax.set_yticks(np.arange(1, 11, step=1))
    ax.set_zlim(-1., 1.)

    surf = ax.plot_surface(X, Y, value_function, cmap=cm.coolwarm, linewidth=0)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(f'fig/{name}.png')

def plot_agent_policy(name, agent):
    fig = plt.figure('Agent policy', figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.set_xticks(np.arange(21))
    ax.set_yticks(np.arange(10))

    ax.set_xticklabels(np.arange(1, 22))
    ax.set_yticklabels(np.arange(1, 11))

    ax.set_xlabel('Player Sum')
    ax.set_ylabel('Dealer showing')

    policy = agent.get_policy()
    image = ax.imshow(policy)
    fig.colorbar(image, shrink=0.5, aspect=5)

    plt.savefig(f'fig/{name}.png')

if __name__ == '__main__':
    agent = MonteCarloAgentEvaluation()
    env = Environment()
    agent.train(100000, env)

    plot_agent_policy('MCEval-policy', agent)
    plot_agent_value_function('MCEval-value-function', agent)


    agent = MonteCarloAgentControl()
    env = Environment()
    agent.train(1000000, env)

    plot_agent_policy('MCControl-policy', agent)
    plot_agent_value_function('MCControl-value-function', agent)
