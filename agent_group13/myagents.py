from random import randint
import ourhexenv

import torch
from torch import nn
import torch.nn.functional as f

import numpy as np
import math

class DQN(nn.Module):
    '''
        Initialization
    '''
    def __init__(self, observation_space, action_space, hidden=256) -> None:
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(observation_space, hidden)  # Connected layer 1
        self.fc2 = nn.Linear(hidden, action_space)  # Connected layer 2

        self.action_counts = np.zeros(action_space)  # UCB Exploration
        self.state_visits = 0  # UCB Current State Visit Count

    '''
        Forward pass
    '''
    def forward(self, state):
        x = f.relu(self.fc1(state))  # ReLU transformation
        return self.fc2(x)
    
    '''
        Epsilon-greedy modified to UCB, where c is UCB's exploration constant
    '''
    def get_valid_action(self, state, board, c=0.8):
        action_probs = self.forward(state)  # Q-value probabilities for each action
        valid_actions = []  # List of valid actions

        # All valid actions in the current board state
        for i in range(len(action_probs)):
            row = i // 11
            col = i % 11

            if row < 11 and col < 11:
                if board[row][col] == 0:  # Empty space
                    valid_actions.append(i)

        if not valid_actions:
            return action_probs.argmax().item()

        valid_action_probs = action_probs[valid_actions]
        #return valid_actions[valid_action_probs.argmax().item()]

        # UCB Implementation
        ucb_values = []

        for action in valid_actions:
            q = valid_action_probs[valid_actions.index(action)].item()
            ucb_term = c * math.sqrt(math.log(self.state_visits + 1)/(self.action_counts[action] +1))
            ucb_values.append(q+ucb_term)

        selected_action = valid_actions[np.argmax(ucb_values)]

        self.action_counts[selected_action] += 1
        self.state_visits +=1

        return selected_action
    

'''
    PLACEHOLDERS FROM P4 FOR TESTING CLASS ENVIRONMENT
'''

class MyDumbAgent():
    def __init__(self, env) -> None:
        self.env = env

    def place(self) -> int:
        xVal = randint(0, self.env.board_size - 1)
        yVal = randint(0, self.env.board_size - 1)

        while self.env.board[xVal][yVal] != 0:
            xVal = randint(0, self.env.board_size - 1)
            yVal = randint(0, self.env.board_size - 1)

        return xVal * self.env.board_size + yVal

    def swap(self) -> int:
        return randint(0,1)

    def select_action(self, observation, reward, termination, truncation, info) -> int:
        return self.place()

class MyABitSmarterAgent():
    def __init__(self, env) -> None:
        self.env = env
        self.visited = set()
        self.start = None

    def place(self) -> int:
        if self.start is None:
            return self.begin()
        temp = self.dfs()
        self.start = (temp // self.env.board_size, temp % self.env.board_size)
        return temp

    def swap(self) -> int:
        return randint(0,1)

    def begin(self) -> int:

        xVal = randint(0, self.env.board_size - 1)
        yVal = randint(0, self.env.board_size - 1)

        while self.env.board[xVal][yVal] != 0:
            xVal = randint(0, self.env.board_size - 1)
            yVal = randint(0, self.env.board_size - 1)

        self.start = (xVal, yVal)
        self.visited.add(self.start)

        return xVal * self.env.board_size + yVal

    def dfs(self) -> int:

        steps = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        temp = [self.start]
        while temp:
            x,y = temp.pop()
        
            for stepX, stepY in steps:
                newX, newY = x + stepX, y + stepY
                if 0 <= newX < self.env.board_size and 0 <= newY < self.env.board_size and self.env.board[newX][newY] == 0:
                    if (newX, newY) not in self.visited:
                        temp.append((newX, newY))
                        self.visited.add((newX,newY))
                        return newX * self.env.board_size + newY

        return self.adj()

    def adj(self) -> int:
        x, y = self.start
        steps = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]

        for stepX, stepY in steps:
            newX, newY = x + stepX, y + stepY
            if 0 <= newX < self.env.board_size and 0 <= newY < self.env.board_size and self.env.board[newX][newY] == 0:
                if (newX, newY) not in self.visited:
                    self.visited.add((newX, newY))
                    return newX * self.env.board_size + newY

        return self.begin()

    def select_action(self, observation, reward, termination, truncation, info) -> int:
        return self.place()


if __name__ == '__main__':
    observation_space = 121
    action_space = 122
    net = DQN(observation_space,action_space)
    state = torch.randn(1, observation_space)
    output = net(state)
    print(output)