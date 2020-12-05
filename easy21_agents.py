import numpy as np
from easy21 import Action, Environment
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
import copy

class MonteCarloAgentEvaluation:
    def __init__(self):
        self.number_visited = np.zeros((10, 21))
        self.total_return = np.zeros((10, 21))
        self.value_function = np.zeros((10, 21))
        self.discount_factor = 0.99
    
    # TODO: not needed
    def get_value_function(self):
        return self.value_function
    
    def get_action(self, state):
        if state.player_sum >= 17:
            return Action.STICK
        else:
            return Action.HIT
    
    def get_policy(self):
        action_lookup = {Action.HIT:0, Action.STICK:1}
        policy = np.zeros((10, 21))
        for i in range(10):
            for j in range(21):
                state = State(dealer_sum=i+1, player_sum=j+1)
                action = self.get_action(state)
                action_value = action_lookup[action]
                policy[i,j] = action_value
        return policy
    
    def predict(self, episode):
        episode_length = len(episode)
        for i in range(episode_length):
            current_return = 0.
            for j in range(i, episode_length):
                reward = episode[j][2]
                current_return += reward * (self.discount_factor ** (j-i))
            state = episode[i][0]
            state_tuple = (state.dealer_sum-1, state.player_sum-1)
            self.total_return[state_tuple] += current_return
            self.value_function[state_tuple] = self.total_return[state_tuple] / self.number_visited[state_tuple]

            
    def train(self, episodes, env):
        for e in range(episodes):
            env.initial_state()
            print(e)
            episode = []
            state = copy.copy(env.state)
            
            while True:
                state_tuple = (state.dealer_sum-1, state.player_sum-1)
                self.number_visited[state_tuple] += 1
                action = self.get_action(state)
                next_state, reward, done = env.step(action)
                print(f'State added: {state}, action: {action}, reward: {reward}')
                episode.append((state, action, reward))
                if done: break
                
                state = next_state
                

            #self.iterations += 1
            if e % 10000 == 0 and e != 0:
                print("Episode: %d" % e)
            
            self.predict(episode) 

        return self.get_value_function()
    

def plot_agent_value_function(agent):
    fig = plt.figure('Agent value function', figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')

    value_function = agent.get_value_function()
    
    X, Y = np.meshgrid(np.arange(1, 22, 1), np.arange(1, 11, 1))
    print(X.shape)
    print(Y.shape)
    print(value_function.shape)

    ax.set_xlabel('Player Sum')
    ax.set_ylabel('Dealer showing')
    ax.set_zlabel('Value')
    ax.set_xticks(np.arange(1, 22, step=5))
    ax.set_yticks(np.arange(1, 11, step=1))
    ax.set_zlim(-1., 1.)

    surf = ax.plot_surface(X, Y, value_function, cmap=cm.coolwarm, linewidth=0)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


if __name__ == '__main__':
    agent = MonteCarloAgentEvaluation()
    env = Environment()
    agent.train(100000, env)

    #plot_agent_policy(agent)
    plot_agent_value_function(agent)