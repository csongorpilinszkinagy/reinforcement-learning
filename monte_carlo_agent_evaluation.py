import numpy as np
import copy

from easy21 import Action, State

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
            action = Action.STICK
        else:
            action =  Action.HIT
        
        state_tuple = (state.dealer_sum-1, state.player_sum-1)
        self.number_visited[state_tuple] += 1
        return action
    
    def get_max_action(self, state):
        return self.get_action(state)
    
    def get_policy(self):
        policy = np.zeros((self.dealer_states, self.player_states))
        for i in range(self.dealer_states):
            for j in range(self.player_states):
                state = State(dealer_sum=i+1, player_sum=j+1)
                policy[i,j] = self.get_max_action(state).value
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