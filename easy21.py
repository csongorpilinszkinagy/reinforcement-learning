from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import copy

class Color(Enum):
    RED  = 0
    BLACK = 1

class Action(Enum):
    HIT = 0
    STICK = 1

class Card:
    def __init__(self, color=None, number=None):
        self.min_card_value = 1
        self.max_card_value = 10
        self.chance_red = 1./3.
        self.chance_black = 1.-self.chance_red
        if number == None:
            self.number = np.random.randint(self.min_card_value, self.max_card_value+1)
        else:
            self.number = number

        if color == None:
            self.color = np.random.choice([Color.RED, Color.BLACK], 1, p=[self.chance_red, self.chance_black])[0]
        else:
            self.color = color

        if self.color==Color.BLACK:
            self.value = self.number
        elif self.color==Color.RED:
            self.value = self.number * -1

class Deck:
    def take_card(self, color=None):
        return Card(color)
    
class State:
    def __init__(self, dealer_sum=0, player_sum=0, terminal=False):
        self.dealer_sum = dealer_sum
        self.player_sum = player_sum
        self.terminal = terminal

        self.bust_min = 1
        self.bust_max = 21
    
    def __str__(self):
        return f'Dealer sum: {self.dealer_sum}, Player sum: {self.player_sum}'

    def add_dealer(self, card):
        self.dealer_sum += card.value
        self.terminal = self.is_terminal()
    
    def add_player(self, card):
        self.player_sum += card.value
        self.terminal = self.is_terminal()
    
    def is_dealer_bust(self):
        if self.dealer_sum < self.bust_min or self.dealer_sum > self.bust_max:
            return True
        return False
    
    def is_player_bust(self):
        if self.player_sum < self.bust_min or self.player_sum > self.bust_max:
            return True
        return False
    
    def is_terminal(self):
        return self.is_dealer_bust() or self.is_player_bust()

class Dealer:
    def __init__(self):
        self.dealer_stop = 17
        # TODO: this should come from the game not from the dealer
        self.bust_min = 1
        self.bust_max = 21

    def get_action(self, state):
        if state.dealer_sum >= self.dealer_stop:
            return Action.STICK
        else:
            return Action.HIT

    def get_policy(self):
        policy = list()
        for i in range(self.bust_min, self.bust_max):
            state = State(dealer_sum=i)
            action = self.get_action(state)
            action_value = Action(action)
            policy.append(action_value)
        return policy

class Environment:
    def __init__(self):
        self.dealer = Dealer()
        self.deck = Deck()
        self.state = State()
        
        
    
    def initial_state(self):
        self.state.dealer_sum = 0
        self.state.player_sum = 0
        self.state.add_dealer(self.deck.take_card(Color.BLACK))
        self.state.add_player(self.deck.take_card(Color.BLACK))

    def dealer_turn(self, state):
        while not state.terminal:
            action = self.dealer.get_action(state)
            if action == Action.HIT:
                card = self.deck.take_card()
                state.add_dealer(card)
            elif action == Action.STICK:
                state.terminal = True
                break
        return state
    
    def step(self, action):
        reward = 0

        if action == Action.HIT:
            self.state.add_player(self.deck.take_card())
            if self.state.is_player_bust():
                reward = -1

        if action == Action.STICK:
            self.state = self.dealer_turn(self.state)
            if self.state.is_dealer_bust():
                reward = 1
            else:
                if self.state.dealer_sum > self.state.player_sum:
                    reward = -1
                if self.state.dealer_sum < self.state.player_sum:
                    reward = 1
        return copy.copy(self.state), reward, self.state.terminal


def plot_dealer_policy():

    fig = plt.figure('Dealer policy', figsize=(10, 5))
    ax = fig.add_subplot(111)

    policy = Dealer().get_policy()
    x = np.arange(1, len(policy)+1)

    ax.plot(x, policy)
    plt.show()

def plot_agent_policy(agent):
    fig = plt.figure('Dealer policy', figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')

    # TODO: replace this mock code with real agent policy plotting
    #policy = agent.get_policy()
    policy = np.zeros((21, 21))
    for i in range(21):
        for j in range(21):
            if j+1 >= 17:
                policy[i, j] = 1
    X, Y = np.meshgrid(np.arange(1, 22), np.arange(1, 22))

    ax.plot_surface(X, Y, policy)
    plt.show()

if __name__ == '__main__':
    env = Environment()
    plot_dealer_policy()
    plot_agent_policy(None)
    assert(False)
    while True:
        action = input()
        if action == 'h':
            action = Action.HIT
        elif action == 's':
            action = Action.STICK
        else:
            continue

        state, reward, done = env.step(action)
        if done:
            break