import numpy as np
from easy21 import Action, Environment
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
import random

class MonteCarloAgentEvaluation:
    def __init__(self):
        self.dealer_states = 10
        self.player_states = 21
        self.number_visited = np.zeros((self.dealer_states, self.player_states))
        self.total_return = np.zeros((self.dealer_states, self.player_states))
        self.value_function = np.zeros((self.dealer_states, self.player_states))
        self.discount_factor = 0.99
        self.player_stop = 17
    
    # TODO: not needed
    def get_value_function(self):
        return self.value_function
    
    def get_action(self, state):
        if state.player_sum >= self.player_stop:
            return Action.STICK
        else:
            return Action.HIT
    
    def get_policy(self):
        policy = np.zeros((self.dealer_states, self.player_states))
        for i in range(self.dealer_states):
            for j in range(self.player_states):
                state = State(dealer_sum=i+1, player_sum=j+1)
                action = self.get_action(state)
                action_value = Action(action)
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
    
class MonteCarloAgentControl:
    def __init__(self):
        self.number_visited = np.zeros((10, 21, 2))
        self.Q = np.zeros((10, 21, 2))
        self.discount_factor = 0.99
    
    def get_alpha(self, state, action):
        return 1.0/(self.number_visited[state.dealer_sum-1][state.player_sum-1][action.value])

    def get_epsilon(self, state):
        return 100./((100. + sum(self.number_visited[state.dealer_sum-1, state.player_sum-1, :]) * 1.0))

    def get_max_Q(self, state):
        np.max(self.Q[state.dealer_sum-1][state.player_sum-1])

    def get_max_action(self, state):
        return Action(np.argmax(self.Q[state.dealer_sum-1][state.player_sum-1]))
    

    def control(self, episode):
        episode_length = len(episode)
        for i in range(episode_length):
            current_return = 0.
            for j in range(i, episode_length):
                reward = episode[j][2]
                current_return += reward * (self.discount_factor ** (j-i))
            state = episode[i][0]
            action = episode[i][1]
            state_action_tuple = (state.dealer_sum-1, state.player_sum-1, action.value)

            self.number_visited[state_action_tuple] += 1

            error = current_return - self.Q[state_action_tuple]
            self.Q[state_action_tuple] += self.get_alpha(state, action) * error

    def policy(self, state):
        #TODO: use numpy instead of if else
        r = random.random()
        if r <= self.get_epsilon(state):
            action = Action(np.random.choice([0,1]))
        else:
            action = self.get_max_action(state)

        return action
    
    def train(self, episodes, env):
        for e in range(episodes):
            env.initial_state()
            print(e)
            episode = []
            state = copy.copy(env.state)
            
            while True:
                state_tuple = (state.dealer_sum-1, state.player_sum-1)
                self.number_visited[state_tuple] += 1
                action = self.policy(state)
                next_state, reward, done = env.step(action)
                print(f'State added: {state}, action: {action}, reward: {reward}')
                episode.append((state, action, reward))
                if done: break
                
                state = next_state
                

            #self.iterations += 1
            if e % 10000 == 0 and e != 0:
                print("Episode: %d" % e)
            
            self.control(episode) 

        return self.get_value_function()
    
    def get_value_function(self):
        return np.max(self.Q, axis=2)
    



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
    agent = MonteCarloAgentControl()
    env = Environment()
    agent.train(10000, env)

    #plot_agent_policy(agent)
    plot_agent_value_function(agent)
