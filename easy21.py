from enum import Enum
import numpy as np

class Color(Enum):
    RED  = 0
    BLACK = 1

class Action(Enum):
    HIT = 0
    STICK = 1

class Card:
    def __init__(self, color=None, number=None):
        if number == None:
            self.number = np.random.randint(1, 10)
        else:
            self.number = number

        if color == None:
            self.color = np.random.choice([Color.RED, Color.BLACK], 1, p=[0.33, 0.67])[0]
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
    
    def __str__(self):
        return f'Dealer sum: {self.dealer_sum}, Player sum: {self.player_sum}'

    def add_dealer(self, card):
        print(f'Dealer card value: {card.value}')
        self.dealer_sum += card.value
        self.terminal = self.is_terminal()
    
    def add_player(self, card):
        print(f'Player card value: {card.value}')
        self.player_sum += card.value
        self.terminal = self.is_terminal()
    
    def is_dealer_bust(self):
        if self.dealer_sum < 1 or self.dealer_sum > 21:
            return True
        return False
    
    def is_player_bust(self):
        if self.player_sum < 1 or self.player_sum > 21:
            return True
        return False
    
    def is_terminal(self):
        return self.is_dealer_bust() or self.is_player_bust()

class Dealer:
    def policy(self, state):
        if state.dealer_sum >= 17:
            return Action.STICK
        else:
            return Action.HIT

class Environment:
    def __init__(self):
        self.dealer = Dealer()
        self.deck = Deck()
        self.state = State()
        
        self.state.add_dealer(self.deck.take_card(Color.BLACK))
        self.state.add_player(self.deck.take_card(Color.BLACK))

    def dealer_turn(self, state):
        while not state.terminal:
            action = self.dealer.policy(state)
            if action == Action.HIT:
                card = self.deck.take_card()
                state.add_dealer(card)
            elif action == Action.STICK:
                state.terminal = True
                break
            print(state)
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
        
        return self.state, reward, self.state.terminal

if __name__ == '__main__':
    env = Environment()
    while True:
        print(env.state)
        action = input()
        if action == 'h':
            action = Action.HIT
        elif action == 's':
            action = Action.STICK
        else:
            continue

        state, reward, done = env.step(action)
        print(f'Reward: {reward}')
        if done:
            break