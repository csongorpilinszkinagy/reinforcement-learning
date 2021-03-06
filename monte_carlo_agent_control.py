import numpy as np
import copy
import random
from easy21 import Action, State

class MonteCarloAgentControl:
    def __init__(self):
        self.dealer_states = 10
        self.player_states = 21
        self.actions = 2
        self.number_visited = np.zeros((self.dealer_states, self.player_states, self.actions))
        self.Q = np.zeros((self.dealer_states, self.player_states, self.actions))
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

            error = current_return - self.Q[state_action_tuple]
            self.Q[state_action_tuple] += self.get_alpha(state, action) * error

    def get_action(self, state):
        #TODO: use numpy instead of if else
        r = random.random()
        if r <= self.get_epsilon(state):
            action = Action(np.random.choice([0,1]))
        else:
            action = self.get_max_action(state)
        
        state_action_tuple = (state.dealer_sum-1, state.player_sum-1, action.value)
        self.number_visited[state_action_tuple] += 1

        return action
    
    # TODO: should be one argmax
    def get_policy(self):
        policy = np.zeros((self.dealer_states, self.player_states))
        for i in range(self.dealer_states):
            for j in range(self.player_states):
                state = State(dealer_sum=i+1, player_sum=j+1)
                policy[i,j] = self.get_max_action(state).value
        return policy
    
    def train(self, episodes, env):
        print('Training Monte Carlo Agent Control')
        for e in range(episodes):
            env.initial_state()
            episode = []
            state = copy.copy(env.state)
            
            while True:
                state_tuple = (state.dealer_sum-1, state.player_sum-1)
                self.number_visited[state_tuple] += 1
                action = self.get_action(state)
                next_state, reward, done = env.step(action)
                episode.append((state, action, reward))
                if done: break
                
                state = next_state
                
            if (e+1) % 10000 == 0:
                print(f'Episode: {e+1}')
            
            self.control(episode) 

        return self.get_value_function()
    
    def get_value_function(self):
        return np.max(self.Q, axis=2)