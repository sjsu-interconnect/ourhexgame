import random
from collections import namedtuple, deque

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    pass

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(float):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, broad_size):
        super(DQN, self).__init__()

        self.broad_size = broad_size
        self.layer1 = nn.Linear(1 + 2 * broad_size ** 2, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, broad_size ** 2 + 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).

    def to_state(self, observation, pie_rule_used, direction):
        # model should be smart enough to figure out a existing 1 / 2 means the slot is occupied
        # need to preprocess the neural net, 1 / 2 on just will not make sense, have to take them apart, treat as two broads≈
        # then, based on direction, arrange the input sequence, see if it's board 1 -> 2 or 2 -> 1
        # should I keep this pure? or add the filters as well
        board1 = np.zeros(self.broad_size ** 2)
        board2 = np.zeros(self.broad_size ** 2)

        # divide into two broads
        observation = observation.flatten()
        for i in range(self.broad_size ** 2):
            stone = observation[i]
            if stone == 0:
                continue
            elif stone == 1:
                board1[i] = 1
            elif stone == 2:
                board2[i] = 1

        nn_input = np.array(pie_rule_used)

        if direction:
            nn_input = np.append(nn_input, board1)
            nn_input = np.append(nn_input, board2)
        else:
            nn_input = np.append(nn_input, board2)
            nn_input = np.append(nn_input, board1)
        return nn_input

    def forward(self, observation=None, pie_rule_used=None, direction=None, processed_state=None):

        if processed_state is not None:
            x = processed_state
        else:
            nn_input = self.to_state(observation, pie_rule_used, direction)
            x = torch.from_numpy(nn_input).float().to(device)

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))

        # filter

        return x

    def pick_action(self, observation, mask, pie_rule_used, direction, processed_state=None):
        x = self.forward(observation, pie_rule_used, direction, processed_state=None)
        q_values = x.cpu().detach().numpy()
        q_values[mask == 0] = -1e9  # Penalize illegal moves
        return np.argmax(q_values)
        # valid_moves = np.multiply(x.cpu().detach().numpy(), mask)
        # return np.argmax(valid_moves)


class G05Agent:
    def __init__(self, env, training_mode=False):
        self.env = env
        self.training_mode = training_mode

        # feel like I should take the observation apart, building a seperated net
        self.board_size = env.board_size
        self.target_net = DQN(self.board_size).to(device)
        self.policy_net = DQN(self.board_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.gamma = 0.99
        self.batch_size = 5
        self.eps_start = 0.9
        self.epsilon = self.eps_start
        self.eps_decay = 0.995
        self.eps_end = 0.05

        self.training_step = 0
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.replay_buffer = ReplayMemory(50)

    def greedy_eps(self, observation, mask, pie_rule_used, direction):
        if random.random() <= self.epsilon:
            self.epsilon = max(self.epsilon * self.eps_decay, self.eps_end)
            choices = []
            for i in range(len(mask)):
                if mask[i] == 1:
                    choices.append(i)
            return random.choice(choices)

        else:
            return self.policy_net.pick_action(observation, mask, pie_rule_used, direction)

    def update(self):
        if self.replay_buffer.__len__() < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, next_states, rewards = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)

        q_values = self.target_net.forward(processed_state=states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.policy_net.forward(processed_state=next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values)

        loss = nn.BCELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        if loss.item() > -80:
            self.save_model()
        if self.training_step % 10 == 0:
            self.target_net = self.policy_net

        self.optimizer.step()

    def train(self, observation, mask, pie_rule_used, direction, reward):
        state = self.policy_net.to_state(observation, pie_rule_used, direction)
        action = self.greedy_eps(observation, mask, pie_rule_used, direction)
        self.env.step(action)
        immediate_reward = self.env.rewards[self.env.agent_selector.next()]

        observation, reward, termination, truncation, info = self.env.last()
        pie_rule_used = observation["pie_rule_used"]
        observation = observation["observation"]
        mask = info['action_mask']
        direction = info["direction"]

        next_state = self.policy_net.to_state(observation, pie_rule_used, direction)
        self.replay_buffer.push(state, action, next_state, immediate_reward)
        state = next_state

        self.training_step += 1
        if self.training_step % 5 == 0:
            self.update()

    def select_action(self, observation, reward, termination, truncation, info):
        pie_rule_used = observation["pie_rule_used"]
        observation = observation["observation"]
        mask = info['action_mask']
        direction = info["direction"]

        if self.training_mode:
            self.train(observation, mask, pie_rule_used, direction, reward)

        self.load_model()
        return self.policy_net.pick_action(observation, mask, pie_rule_used, direction)  # or load the NN from somewhere

    def load_model(self, PATH="policy_net.pth"):
        self.policy_net.load_state_dict(torch.load(PATH))

    def save_model(self, PATH="policy_net.pth"):
        torch.save(self.policy_net.state_dict(), PATH)
