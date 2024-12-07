import numpy as np

from ourhexenv import OurHexGame

# Hyperparameters
BOARD_SIZE = 11
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_MEMORY_SIZE = 10000
NUM_CNN_CHANNELS = 64
LSTM_HIDDEN_SIZE = 256
FC_HIDDEN_SIZE = 128  # size of layer between the last fully connected layer and output layer.
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
    """
    Class defines the Neural Network architecture which the agent will use
    """
    def __init__(self, board_size, num_channels=NUM_CNN_CHANNELS):
        super(HexNet, self).__init__()
        self.board_size = board_size

        # CNN layers for pattern recognition.
        self.conv1 = nn.Conv2d(1, num_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, padding=1)

        # batch norm layers to come after the CNN layers.
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)

        # LSTM layer to attempt to capture step/time related patterns.
        self.lstm = nn.LSTM(
            input_size=num_channels * board_size * board_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        # fully connected linear layers for output.
        self.fc1 = nn.Linear(LSTM_HIDDEN_SIZE, FC_HIDDEN_SIZE)
        self.fc2 = nn.Linear(FC_HIDDEN_SIZE, board_size * board_size + 1)

    def forward(self, x):
        """
        feed forward function to get the output from the network.
        :param x: input values (board observation)
        :return: output probability distribution
        """
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
    """
    Replay Memory to aid in training process
    """
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


class G01Agent:
    def __init__(self, board_size, device=torch.device('cpu'), load_checkpoint=None):
        """
        Initialize the agent.
        :param board_size: board size from the environment.
        :param device: device for pytorch, cpu if nothing, use GPU if available.
        :param load_checkpoint: path to model weights if desired.
        """
        self.device = device
        self.board_size = board_size
        self.policy_net = HexNet(board_size).to(device)
        self.target_net = HexNet(board_size).to(device)

        # action counts for ucb1
        self.action_counts = torch.zeros(board_size * board_size + 1, device=device)

        # Create optimizer before loading checkpoint
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(),
                                           lr=LEARNING_RATE, amsgrad=True)

        if load_checkpoint:
            self.load_checkpoint(load_checkpoint)
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.memory = ReplayMemory(REPLAY_MEMORY_SIZE)
        self.steps_done = 0
        self.episodes_done = 0

        # make directories for checkpoints.
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    def save_checkpoint(self, episode=None):
        """Save model checkpoint with current episode number"""
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'action_counts': self.action_counts
        }

        filename = f'checkpoint_episode_{episode if episode else self.episodes_done}.pt'
        path = os.path.join(CHECKPOINT_DIR, filename)
        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']
        self.action_counts = checkpoint['action_counts']

    # this is a hidden helper function because i am using this for training and the actual select_action for use in the competition
    def _select_action(self, state, action_mask):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)

            # Calculate UCB values for each action
            ucb_values = q_values.clone()
            valid_actions = torch.where(torch.tensor(action_mask, device=self.device) == 1)[0]

            # For valid actions, calculate UCB term
            for action in valid_actions:
                if self.action_counts[action] > 0:
                    ucb_term = math.sqrt(2 * math.log(self.steps_done) / self.action_counts[action])
                    ucb_values[0, action] += ucb_term
                else:
                    ucb_values[0, action] = float('inf')  # Ensure unvisited actions are tried

            # Mask invalid moves
            mask = torch.where(
                torch.tensor(action_mask, device=self.device) == 1,
                torch.zeros_like(ucb_values),
                torch.ones_like(ucb_values) * float('-inf')
            )
            ucb_values = ucb_values + mask

            # Select action with highest UCB value
            action = ucb_values.max(1)[1].view(1, 1)

            # update our action in the action counts table
            self.action_counts[action.item()] += 1

            return action

    def select_action(self, observation, reward, termination, truncation, info):
        board = observation['observation']
        action_mask = info['action_mask']

        return self._select_action(board, action_mask)


def optimize_model(agent, batch_size=BATCH_SIZE, gamma=GAMMA):
    # Skip if we don't have enough samples in memory
    if len(agent.memory) < batch_size:
        return

    # Sample a batch of transitions from memory
    transitions = agent.memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043)
    batch = Transition(*zip(*transitions))

    # Create mask for non-final states (where next_state is not None)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=agent.device, dtype=torch.bool)

    # Convert batch arrays to tensors and move to device
    state_batch = torch.FloatTensor(np.array(batch.state)).to(agent.device)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    state_action_values = agent.policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states
    next_state_values = torch.zeros(batch_size, device=agent.device)
    with torch.no_grad():  # No need to compute gradients for next state values
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(agent.device)
        # Get max predicted Q values for next states from target network
        next_state_values[non_final_mask] = agent.target_net(next_state_batch).max(1)[0]

    # Compute the expected Q values: reward + gamma * max(Q(s_{t+1}))
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss between current Q values and expected Q values
    loss = F.smooth_l1_loss(state_action_values,
                           expected_state_action_values.unsqueeze(1))

    # Optimize the model
    agent.optimizer.zero_grad()  # Clear previous gradients
    loss.backward()  # Compute gradients
    torch.nn.utils.clip_grad_value_(agent.policy_net.parameters(), 100)  # Clip gradients to prevent explosion
    agent.optimizer.step()  # Update weights


def train_agent(env, agent, num_episodes):
    for episode in range(num_episodes):
        # Reset environment at start of episode
        env.reset()
        observation, rewards, terminations, truncations, infos = env.last()
        total_reward = 0
        done = False
        print(f"Completed episode {episode}/{num_episodes}")

        while not done:
            # Select and perform an action
            action = agent.select_action(
                observation['observation'],
                infos['action_mask']
            )
            env.step(action.item())
            next_observation, rewards, terminations, truncations, infos = env.last()

            # Store transition in memory
            reward_tensor = torch.tensor([rewards], device=agent.device)
            agent.memory.push(
                observation['observation'],
                action,
                next_observation['observation'],
                reward_tensor,
                terminations
            )

            # Move to next state
            observation = next_observation
            total_reward += rewards
            done = terminations or truncations

            # Perform one step of optimization
            optimize_model(agent, BATCH_SIZE, GAMMA)

            # Update target network periodically
            if agent.steps_done % TARGET_UPDATE_FREQ == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

        # Episode complete
        agent.episodes_done += 1
        if episode % CHECKPOINT_FREQ == 0:
            agent.save_checkpoint()

    # Save final model
    agent.save_checkpoint()


if __name__ == '__main__':
    training_env = OurHexGame(11, False, "human")
    compute_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent_to_train = G01Agent(11, compute_device, "checkpoints/checkpoint_episode_8192.pt")
    train_agent(training_env, agent_to_train, num_episodes=8)
    # add
