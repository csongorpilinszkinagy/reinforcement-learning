import numpy as np
from easy21 import Action, Environment
import copy
import matplotlib.pyplot as plt

class MonteCarloAgent:
    def __init__(self):
        self.number_visited = np.zeros((21, 21))
        self.total_return = np.zeros((21, 21))
        self.value_function = np.zeros((21, 21))
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
        policy = np.zeros((21, 21))
        for i in range(21):
            for j in range(21):
                state = State(dealer_sum=i, player_sum=j)
                action = self.get_action(state)
                action_value = action_lookup[action]
                policy[i,j] = action_value
        return policy
    
    def predict(self, episode):
        print(episode)
        #print(episode[-1][0])
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
            state = env.state
            
            while True:
                print(state)
                state_tuple = (state.dealer_sum-1, state.player_sum-1)
                self.number_visited[state_tuple] += 1
                action = self.get_action(state)
                next_state, reward, done = env.step(action)
                if done: break
                print(f'State added: {state}, action: {action}, reward: {reward}')
                episode.append((copy.copy(state), action, reward))
                
                state = next_state
                

            #self.iterations += 1
            if e % 10000 == 0 and e != 0:
                print("Episode: %d" % e)
            
            self.predict(episode) 

        return self.get_value_function()

def plot_agent_policy(agent):
    fig = plt.figure('Agent policy', figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')

    #policy = agent.get_policy()
    policy = np.zeros((21, 21))
    for i in range(21):
        for j in range(21):
            if j+1 >= 17:
                policy[i, j] = 1
    X, Y = np.meshgrid(np.arange(1, 22), np.arange(1, 22))

    ax.plot_surface(X, Y, policy)
    plt.show()

def plot_agent_value_function(agent):
    fig = plt.figure('Agent value function', figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')

    value_function = agent.get_value_function()
    X, Y = np.meshgrid(np.arange(1, 22), np.arange(1, 22))

    ax.plot_surface(X, Y, value_function)
    plt.show()


if __name__ == '__main__':
    agent = MonteCarloAgent()
    env = Environment()
    agent.train(10000, env)

    plot_agent_policy(agent)
    plot_agent_value_function(agent)
    


