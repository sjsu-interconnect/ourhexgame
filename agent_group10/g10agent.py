import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import yaml
import os

class EnhancedDuelingDQN(nn.Module):
    """
    Enhanced Dueling DQN with CNN layers for HexGame.
    """
    def __init__(self, state_size, action_size, board_size):
        super(EnhancedDuelingDQN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        flattened_size = board_size * board_size * 128
        self.shared_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, state):
        state = state.unsqueeze(1)  # Add channel dimension
        x = self.conv_layers(state)
        shared_output = self.shared_layer(x)
        value = self.value_stream(shared_output)
        advantages = self.advantage_stream(shared_output)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values



def load_hyperparameters(config_file, config_key):
    """Load hyperparameters from a YAML file."""
    # Resolve the absolute path of the hyperparameters file
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Hyperparameters file not found at: {config_path}")
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config[config_key]

this_path = os.path.abspath(__file__)
my_dir = os.path.dirname(this_path)
model_path = os.path.join(my_dir, "g10_model.pt")



class G10Agent:
    """
    Optimized RL Agent for HexGame with full functionality.
    """
    def __init__(self, env, config_file="hyperparameters.yml", config_key="default", model_save_path=model_path):
        # Load hyperparameters
        self.params = load_hyperparameters(config_file, config_key)

        self.env = env
        self.state_size = np.prod(env.board.shape)  # Flattened board state
        self.action_size = env.board_size * env.board_size + 1
        self.board_size = env.board_size
        self.sparse_rewards = self.params["sparse_rewards"]

        # Hyperparameters
        self.gamma = self.params["gamma"]
        self.epsilon = self.params["epsilon"]
        self.epsilon_min = self.params["epsilon_min"]
        self.epsilon_decay = self.params["epsilon_decay"]
        self.learning_rate = self.params["learning_rate"]
        self.batch_size = self.params["batch_size"]
        self.memory_size = self.params["memory_size"]
        self.target_update_freq = self.params["target_update_freq"]

        # Replay memory
        self.memory = deque(maxlen=self.memory_size)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural network models
        self.model = EnhancedDuelingDQN(self.state_size, self.action_size, self.board_size).to(self.device)
        self.target_model = EnhancedDuelingDQN(self.state_size, self.action_size, self.board_size).to(self.device)
        self.update_target_model()

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()

        # Training counters
        self.update_counter = 0
        self.model_save_path = model_save_path

        # Load model if available
        self.load(self.model_save_path)

    def update_target_model(self):
        """Soft update of target network parameters."""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, observation, reward, termination, truncation, info):
        """Epsilon-greedy action selection."""
        board_state = torch.FloatTensor(observation["observation"]).unsqueeze(0).to(self.device)
        valid_actions = [
            action for action in range(self.action_size - 1)
            if observation["observation"].flatten()[action] == 0
        ]
        if info.get("pie_rule_allowed", False):
            valid_actions.append(self.action_size - 1)

        if not valid_actions:
            raise ValueError("No valid actions available.")

        if np.random.rand() <= self.epsilon:
            action = random.choice(valid_actions)
        else:
            with torch.no_grad():
                q_values = self.model(board_state).cpu().numpy().flatten()
            action = valid_actions[np.argmax(q_values[valid_actions])]

        return action

    def replay(self):
        """Train the model using replay memory."""
        if len(self.memory) < self.batch_size:
            return

        # Sample minibatch
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute current Q-values
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute loss
        loss = self.criterion(q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_model()

    def save(self):
        """Save the current model to disk."""
        print(f"Saving model to {self.model_save_path}")
        torch.save(self.model.state_dict(), self.model_save_path)

    def load(self, filename):
        """Load a saved model if it exists."""
        try:
            self.model.load_state_dict(torch.load(filename))
            self.update_target_model()
            print(f"Loaded model from {filename}")
        except FileNotFoundError:
            print(f"No saved model found at {filename}. Starting fresh.")

    def switch_reward_env(self, sparse_rewards):
        """Switch between sparse and dense reward environments."""
        self.sparse_rewards = sparse_rewards
        print(f"Switched to {'sparse' if sparse_rewards else 'dense'} rewards.")
