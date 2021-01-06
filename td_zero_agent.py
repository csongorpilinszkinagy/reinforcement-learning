import numpy as np
import copy

from easy21 import Action

class TDZeroAgent:
    def __init__(self):
        self.lambda_value = 1
        self.E = np.zeros((10, 21, 2))
        self.N = np.zeros((10, 21, 2))
        self.Q = np.zeros((10, 21, 2))
        self.epsilon=0.01
        self.discount_factor = 0.99

        return

    def get_Q(self, state, action):
        print(state, action)
        return self.Q[state.dealer_sum-1, state.player_sum-1, action.value]

    def get_alpha(self, state, action):
        return 1./self.N[state.dealer_sum-1, state.player_sum-1, action.value]
    
    def get_epsilon(self, state):
        return 100/((100 + sum(self.N[state.dealer_sum-1, state.player_sum-1, :]) * 1.0))
    
    def get_best_action(self, state):
        action = Action(np.argmax(self.Q[state.dealer_sum-1, state.player_sum-1, :]))
        return action

    def get_action(self, state):
        epsilon = self.get_epsilon(state)
        best_action = self.get_best_action(state)
        random_action = np.random.choice([Action.HIT, Action.STICK], 1)[0]
        action = np.random.choice([best_action, random_action], 1, [1.-epsilon, epsilon])[0]

        state_action_tuple = (state.dealer_sum-1, state.player_sum-1, action.value)
        self.N[state_action_tuple] += 1

        return action
    
    def get_policy(self):
        return np.argmax(self.Q, axis=2)
    
    def get_value_function(self):
        return np.max(self.Q, axis=2)

    
    def train(self, steps, env):
        env.initial_state()
        state = copy.copy(env.state)
        print(state)
        action = self.get_action(state)

        next_action = action

        while True:
            print(state, action)
            next_state, reward, done = env.step(action)
            q = self.get_Q(state, action)

            if not next_state.terminal:
                next_action = self.get_action(next_state)
                q_next = self.get_Q(next_state, next_action)
                delta = reward + (q_next - q) * self.lambda_value
            else:
                delta = reward - q * self.lambda_value

            self.E[state.dealer_sum-1, state.player_sum-1, action.value] += 1

            alpha = self.get_alpha(state, action)
            Q_update = alpha * delta * self.E
            self.Q += Q_update
            self.E *= self.discount_factor * self.lambda_value

            if done:
                break

            state = next_state
            action = next_action
