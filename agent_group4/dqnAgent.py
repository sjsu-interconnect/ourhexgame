# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from collections import deque

# # -------------------------------
# # DQNNetwork for smartAgent (outputs Q-values)
# # -------------------------------
# class DQNNetwork(nn.Module):
#     def __init__(self, board_size, num_actions):
#         super(DQNNetwork, self).__init__()
#         self.board_size = board_size
#         self.num_actions = num_actions

#         # Convolutional layers
#         self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)

#         # Fully connected layers
#         self.fc1 = nn.Linear(128 * board_size * board_size, 256)
#         self.fc2 = nn.Linear(256, num_actions)

#     def forward(self, x):
#         x = torch.relu(self.bn1(self.conv1(x)))
#         x = torch.relu(self.bn2(self.conv2(x)))
#         x = torch.relu(self.bn3(self.conv3(x)))
#         x = x.view(x.size(0), -1)
#         x = torch.relu(self.fc1(x))
#         q_values = self.fc2(x)
#         return q_values


# # -------------------------------
# # ReplayBuffer class
# # -------------------------------
# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)

#     def add(self, experience):
#         self.buffer.append(experience)

#     def sample(self, batch_size):
#         return random.sample(self.buffer, batch_size)

#     def __len__(self):
#         return len(self.buffer)
    
# class DQNAgent:
#     def __init__(self, env, board_size=11, learning_rate=1e-4, gamma=0.99,
#                  buffer_capacity=50000, batch_size=64, target_update_freq=1000,
#                  tau=1.0):
#         self.env = env
#         self.board_size = board_size
#         self.num_actions = board_size * board_size + 1  # Including pie rule if applicable
#         self.gamma = gamma
#         self.batch_size = batch_size
#         self.tau = tau  # For soft updates
#         self.target_update_freq = target_update_freq

#         # Initialize Networks
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.policy_net = DQNNetwork(board_size, self.num_actions).to(self.device)
#         self.target_net = DQNNetwork(board_size, self.num_actions).to(self.device)
#         self.target_net.load_state_dict(self.policy_net.state_dict())
#         self.target_net.eval()  # Target network is not trained

#         # Optimizer
#         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

#         # Replay Buffer
#         self.replay_buffer = ReplayBuffer(buffer_capacity)

#         # Initialize steps
#         self.steps_done = 0

#     def preprocess_observation(self, observation_dict, agent):
#         """
#         Preprocess the observation to create a tensor input for the neural network.

#         Args:
#             observation_dict (dict): The observation returned by the environment's observe() method.
#             agent (str): The current agent ('1' or 'player_2').

#         Returns:
#             torch.Tensor: The preprocessed observation tensor.
#         """
#         board = observation_dict["observation"]  # Extract the board (NumPy array)
#         pie_rule_used = observation_dict["pie_rule_used"]  # Extract pie rule usage (0 or 1)

#         # Create feature channels
#         player1 = (board == 1).astype(np.float32)  # Player 1's positions
#         player2 = (board == 2).astype(np.float32)  # Player 2's positions
#         current_player = np.full((self.board_size, self.board_size), 1.0 if agent == "player_1" else 0.0, dtype=np.float32)
#         pie_rule_channel = np.full((self.board_size, self.board_size), float(pie_rule_used), dtype=np.float32)

#         # Stack channels into a single input tensor
#         state = np.stack([player1, player2, current_player, pie_rule_channel], axis=0)  # Shape: (4, board_size, board_size)
#         state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)  # Shape: (1, 4, board_size, board_size)

#         return state_tensor

#     def select_action(self, observation, agent):
#         """
#         Select an action using Boltzmann (Softmax) exploration strategy.

#         Args:
#             observation (dict): The observation dictionary from the environment.
#             agent (str): The current agent ('player_1' or 'player_2').

#         Returns:
#             int: The selected action.
#         """
#         state = self.preprocess_observation(observation, agent)
#         with torch.no_grad():
#             q_values = self.policy_net(state)  # Shape: (1, num_actions)
#             q_values = q_values.cpu().numpy().flatten()

#         # Apply Boltzmann (Softmax) exploration
#         tau = 0.5  # Temperature parameter, can be decayed over time
#         probabilities = np.exp(q_values / tau)

#         # Mask invalid actions
#         valid_actions = self.get_valid_actions(observation)
#         probabilities *= valid_actions  # Zero out invalid actions

#         # Handle edge case: If all probabilities are zero (e.g., no valid actions)
#         if probabilities.sum() == 0 or np.isnan(probabilities).any():
#             print("Invalid probabilities detected. Defaulting to random valid action.")
#             valid_indices = np.where(valid_actions == 1)[0]
#             return np.random.choice(valid_indices)

#         # Normalize probabilities
#         probabilities /= probabilities.sum()  # Re-normalize after masking

#         # Log probabilities for debugging
#         #print(f"Normalized probabilities: {probabilities}")

#         # Choose an action based on the probabilities
#         action = np.random.choice(self.num_actions, p=probabilities)
#         return action

#     def get_valid_actions(self, observation):
#         """
#         Returns a binary mask of valid actions.

#         Args:
#             observation (dict): The observation dictionary from the environment.

#         Returns:
#             np.ndarray: A 1D array where 1 indicates a valid action and 0 indicates invalid.
#         """
#         board = observation["observation"]
#         pie_rule_used = observation["pie_rule_used"]

#         # Initialize valid actions mask
#         valid = (board == 0).astype(np.float32).flatten()

#         # Handle pie rule (action index = board_size * board_size)
#         pie_rule_valid = 1.0 if (not self.env.is_pie_rule_used and self.env.agent_selection == "player_2") else 0.0
#         valid = np.append(valid, pie_rule_valid)

#         return valid

#     def store_experience(self, state, action, reward, next_state, done):
#         """
#         Stores the experience in the replay buffer.

#         Args:
#             state (torch.Tensor): Current state tensor.
#             action (int): Action taken.
#             reward (float): Reward received.
#             next_state (torch.Tensor): Next state tensor.
#             done (bool): Whether the episode has ended.
#         """
#         self.replay_buffer.add((state, action, reward, next_state, done))

#     def train_step(self):
#         """
#         Performs a single training step.
#         """
#         if len(self.replay_buffer) < self.batch_size:
#             return

#         batch = self.replay_buffer.sample(self.batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)

#         # Convert to tensors
#         states = torch.cat(states).to(self.device)           # Shape: (batch, 4, board_size, board_size)
#         actions = torch.tensor(actions, dtype=torch.long).to(self.device)  # Shape: (batch,)
#         rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)  # Shape: (batch,)
#         next_states = torch.cat(next_states).to(self.device)  # Shape: (batch, 4, board_size, board_size)
#         dones = torch.tensor(dones, dtype=torch.float32).to(self.device)      # Shape: (batch,)

#         # Current Q values
#         current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # Shape: (batch,)

#         # Next Q values from target network
#         with torch.no_grad():
#             next_q = self.target_net(next_states).max(1)[0]  # Shape: (batch,)

#         # Compute target Q values
#         target_q = rewards + (self.gamma * next_q * (1 - dones))

#         # Compute loss (Mean Squared Error)
#         loss = nn.MSELoss()(current_q, target_q)

#         # Optimize the model
#         self.optimizer.zero_grad()
#         loss.backward()
#         # Clip gradients to prevent explosion
#         nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
#         self.optimizer.step()

#         # Optionally update the target network
#         self.steps_done += 1
#         if self.steps_done % self.target_update_freq == 0:
#             self.update_target_network()

#     def update_target_network(self):
#         """
#         Updates the target network by copying weights from the policy network.
#         """
#         self.target_net.load_state_dict(self.policy_net.state_dict())

# # env = OurHexGame(board_size=11)
# # env.reset()

# def load_dqn_agent(env, filepath):
#     agent = DQNAgent(env)
#     agent.policy_net.load_state_dict(torch.load(filepath))
#     agent.policy_net.eval()  # Set to evaluation mode

#     return agent


# # Example usage
# # loaded_dqn_agent = load_dqn_agent(env, "dqn_agent.pt")

# Filename: dqnAgent.py

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from math import sqrt
from ourhexenv import OurHexGame  # Ensure this import is correct based on your project structure

# -------------------------------
# DQNNetwork for smartAgent (outputs Q-values)
# -------------------------------
class DQNNetwork(nn.Module):
    def __init__(self, board_size, num_actions):
        super(DQNNetwork, self).__init__()
        self.board_size = board_size
        self.num_actions = num_actions

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * board_size * board_size, 256)
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values


# -------------------------------
# ReplayBuffer class
# -------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self, env, board_size=11, learning_rate=1e-4, gamma=0.99,
                 buffer_capacity=50000, batch_size=64, target_update_freq=1000,
                 tau=1.0):
        self.env = env
        self.board_size = board_size
        self.num_actions = board_size * board_size + 1  # Including pie rule if applicable
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau  # For soft updates
        self.target_update_freq = target_update_freq

        # Initialize Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork(board_size, self.num_actions).to(self.device)
        self.target_net = DQNNetwork(board_size, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is not trained

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Initialize steps
        self.steps_done = 0

    def preprocess_observation(self, observation_dict, agent):
        """
        Preprocess the observation to create a tensor input for the neural network.

        Args:
            observation_dict (dict): The observation returned by the environment's observe() method.
            agent (str): The current agent ('1' or 'player_2').

        Returns:
            torch.Tensor: The preprocessed observation tensor.
        """
        board = observation_dict["observation"]  # Extract the board (NumPy array)
        pie_rule_used = observation_dict["pie_rule_used"]  # Extract pie rule usage (0 or 1)

        # Create feature channels
        player1 = (board == 1).astype(np.float32)  # Player 1's positions
        player2 = (board == 2).astype(np.float32)  # Player 2's positions
        current_player = np.full((self.board_size, self.board_size), 1.0 if agent == "player_1" else 0.0, dtype=np.float32)
        pie_rule_channel = np.full((self.board_size, self.board_size), float(pie_rule_used), dtype=np.float32)

        # Stack channels into a single input tensor
        state = np.stack([player1, player2, current_player, pie_rule_channel], axis=0)  # Shape: (4, board_size, board_size)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)  # Shape: (1, 4, board_size, board_size)

        return state_tensor

    def select_action(self, observation, agent):
        """
        Select an action using Boltzmann (Softmax) exploration strategy.

        Args:
            observation (dict): The observation dictionary from the environment.
            agent (str): The current agent ('player_1' or 'player_2').

        Returns:
            int: The selected action.
        """
        state = self.preprocess_observation(observation, agent)
        with torch.no_grad():
            q_values = self.policy_net(state)  # Shape: (1, num_actions)
            q_values = q_values.cpu().numpy().flatten()

        # Apply Boltzmann (Softmax) exploration with numerical stability
        tau = 0.5  # Temperature parameter, can be decayed over time
        q_values = q_values / tau
        q_values -= np.max(q_values)  # For numerical stability
        probabilities = np.exp(q_values)
        sum_prob = probabilities.sum()

        if sum_prob == 0:
            # If sum is zero, fallback to uniform probabilities over valid actions
            valid_actions = self.get_valid_actions(observation)
            if valid_actions.sum() == 0:
                raise ValueError("No valid actions available.")
            probabilities = valid_actions.copy()
            probabilities /= probabilities.sum()
        else:
            probabilities /= sum_prob

            # Mask invalid actions
            valid_actions = self.get_valid_actions(observation)
            probabilities *= valid_actions
            sum_prob = probabilities.sum()
            if sum_prob == 0:
                # If no valid actions after masking, fallback to uniform over valid actions
                probabilities = valid_actions.copy()
                probabilities /= probabilities.sum()
            else:
                probabilities /= sum_prob

        # Now, probabilities should be a valid distribution
        action = np.random.choice(self.num_actions, p=probabilities)
        return action


    def get_valid_actions(self, observation):
        """
        Returns a binary mask of valid actions.

        Args:
            observation (dict): The observation dictionary from the environment.

        Returns:
            np.ndarray: A 1D array where 1 indicates a valid action and 0 indicates invalid.
        """
        board = observation["observation"]
        pie_rule_used = observation["pie_rule_used"]

        # Initialize valid actions mask
        valid = (board == 0).astype(np.float32).flatten()

        # Handle pie rule (action index = board_size * board_size)
        pie_rule_valid = 1.0 if (not self.env.is_pie_rule_used and self.env.agent_selection == "player_2") else 0.0
        valid = np.append(valid, pie_rule_valid)

        return valid

    def store_experience(self, state, action, reward, next_state, done):
        """
        Stores the experience in the replay buffer.

        Args:
            state (torch.Tensor): Current state tensor.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (torch.Tensor): Next state tensor.
            done (bool): Whether the episode has ended.
        """
        self.replay_buffer.add((state, action, reward, next_state, done))

    def train_step(self):
        """
        Performs a single training step.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.cat(states).to(self.device)           # Shape: (batch, 4, board_size, board_size)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)  # Shape: (batch,)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)  # Shape: (batch,)
        next_states = torch.cat(next_states).to(self.device)  # Shape: (batch, 4, board_size, board_size)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)      # Shape: (batch,)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # Shape: (batch,)

        # Next Q values from target network
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]  # Shape: (batch,)

        # Compute target Q values
        target_q = rewards + (self.gamma * next_q * (1 - dones))

        # Compute loss (Mean Squared Error)
        loss = nn.MSELoss()(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent explosion
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Optionally update the target network
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.update_target_network()

    def update_target_network(self):
        """
        Updates the target network by copying weights from the policy network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

def load_dqn_agent(env, filepath):
    agent = DQNAgent(env)
    agent.policy_net.load_state_dict(torch.load(filepath))
    agent.policy_net.eval()  # Set to evaluation mode

    return agent


# Example usage
# loaded_dqn_agent = load_dqn_agent(env, "dqn_agent.pt")