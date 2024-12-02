# Hyperparameters
import numpy as np

from ourhexenv import OurHexGame

BOARD_SIZE = 11
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_MEMORY_SIZE = 10000
NUM_CNN_CHANNELS = 64
LSTM_HIDDEN_SIZE = 256
FC_HIDDEN_SIZE = 128
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TARGET_UPDATE_FREQ = 10
CHECKPOINT_FREQ = 1000
CHECKPOINT_DIR = 'checkpoints'

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import random
import os
from collections import deque, namedtuple

Transition = namedtuple('Transition',
    ('state', 'action', 'next_state', 'reward', 'terminated'))

class HexNet(nn.Module):
    def __init__(self, board_size, num_channels=NUM_CNN_CHANNELS):
        super(HexNet, self).__init__()
        self.board_size = board_size

        # CNN layers for pattern recognition
        self.conv1 = nn.Conv2d(1, num_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)

        self.lstm = nn.LSTM(
            input_size=num_channels * board_size * board_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.fc1 = nn.Linear(LSTM_HIDDEN_SIZE, FC_HIDDEN_SIZE)
        self.fc2 = nn.Linear(FC_HIDDEN_SIZE, board_size * board_size + 1)

    def forward(self, x):
        batch_size = x.size(0)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(batch_size, 1, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward, terminated):
        """Save a transition"""
        self.memory.append(Transition(state, action, next_state, reward, terminated))

    def sample(self, batch_size):
        """Random sample a batch of transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class HexAgent:
    def __init__(self, board_size, device, load_checkpoint=None):
        self.device = device
        self.board_size = board_size
        self.policy_net = HexNet(board_size).to(device)
        self.target_net = HexNet(board_size).to(device)

        if load_checkpoint:
            self.load_checkpoint(load_checkpoint)
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(),
                                           lr=LEARNING_RATE, amsgrad=True)
        self.memory = ReplayMemory(REPLAY_MEMORY_SIZE)

        self.steps_done = 0
        self.episodes_done = 0

        # Create checkpoint directory if it doesn't exist
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    def save_checkpoint(self, episode=None):
        """Save model checkpoint with current episode number"""
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done
        }

        filename = f'checkpoint_episode_{episode if episode else self.episodes_done}.pt'
        path = os.path.join(CHECKPOINT_DIR, filename)
        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']

    def select_action(self, state, action_mask):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                mask = torch.where(
                    torch.tensor(action_mask, device=self.device) == 1,
                    torch.zeros_like(q_values),
                    torch.ones_like(q_values) * float('-inf')
                )
                q_values = q_values + mask
                return q_values.max(1)[1].view(1, 1)
        else:
            # Get valid actions from action mask
            valid_actions = np.where(action_mask == 1)[0]
            return torch.tensor([[random.choice(valid_actions)]],
                                device=self.device, dtype=torch.long)

    def select_action(self, observation, ):


def optimize_model(agent, batch_size=32, gamma=0.99):
    if len(agent.memory) < batch_size:
        return

    transitions = agent.memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=agent.device, dtype=torch.bool)

    state_batch = torch.FloatTensor(np.array(batch.state)).to(agent.device)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)  # Now this will work since rewards are tensors

    state_action_values = agent.policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=agent.device)
    with torch.no_grad():
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(agent.device)
        next_state_values[non_final_mask] = agent.target_net(next_state_batch).max(1)[0]

    expected_state_action_values = (next_state_values * gamma) + reward_batch

    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

    agent.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(agent.policy_net.parameters(), 100)
    agent.optimizer.step()


def train_agent(env, agent, num_episodes):
    for episode in range(num_episodes):
        env.reset()
        observation, rewards, terminations, truncations, infos = env.last()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(
                observation['observation'],
                infos['action_mask']
            )
            env.step(action.item())
            next_observation, rewards, terminations, truncations, infos = env.last()

            # Convert reward to tensor when pushing to memory
            reward_tensor = torch.tensor([rewards], device=agent.device)
            agent.memory.push(
                observation['observation'],
                action,
                next_observation['observation'],
                reward_tensor,  # Now pushing a tensor instead of int
                terminations
            )

            observation = next_observation
            total_reward += rewards
            done = terminations or truncations

            optimize_model(agent, BATCH_SIZE, GAMMA)

            if agent.steps_done % TARGET_UPDATE_FREQ == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

        agent.episodes_done += 1
        if episode % CHECKPOINT_FREQ == 0:
            agent.save_checkpoint()

    agent.save_checkpoint()


if __name__ == '__main__':
    env = OurHexGame(11, False, "human")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = HexAgent(11, device)
    train_agent(env, agent, num_episodes=64)