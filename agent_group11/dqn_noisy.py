import random
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import torch.nn.functional as fn

# Noisy Neural Network Module, used to introduce stochasticity in the exploration strategy
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize parameters for mean and standard deviation of weights and biases
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        
        self.std_init = std_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # Initialize weights and biases using a uniform distribution
        mu_range = 1 / self.in_features**0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / self.in_features**0.5)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / self.out_features**0.5)

    def reset_noise(self):
        # Generate random noise for weights and biases
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, input):
        if self.training:  # Add noise during training
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:  # Use deterministic weights during evaluation
            weight = self.weight_mu
            bias = self.bias_mu
        return fn.linear(input, weight, bias)

# Replay Buffer to store transitions for experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # Use deque to maintain a fixed-size buffer

    def push(self, state, action, reward, next_state, done):
        # Store a single transition in the buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Randomly sample a batch of transitions
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)  # Return the current size of the buffer

# Noisy DQN architecture with multiple hidden layers
class DQN_noisy(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN_noisy, self).__init__()
        
        # Define the neural network layers
        self.fc1 = nn.Linear(np.prod(input_shape), 128)
        self.fc2 = NoisyLinear(128, 128)
        self.fc3 = NoisyLinear(128, 256)
        self.fc4 = NoisyLinear(256, 512)
        self.fc5 = NoisyLinear(512, 256)
        self.fc6 = NoisyLinear(256, 128)
        self.fc7 = NoisyLinear(128, num_actions)

    def forward(self, x):
        # Forward pass through the network
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        return self.fc7(x)

# DQN Agent with training and action selection methods
class DQNAgent:
    def __init__(self, env, buffer_size=10000, batch_size=128, gamma=0.99, lr=1e-3, name='player_1', policy_file_path=None):
        self.env = env
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.name = name
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Extract input shape and number of actions from the environment
        input_shape = env.observation_spaces[env.possible_agents[0]]["observation"].shape
        num_actions = env.action_spaces[env.possible_agents[0]].n

        # Initialize policy and target networks
        self.policy_net = DQN_noisy(input_shape, num_actions).to(self.device)
        self.target_net = DQN_noisy(input_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set the target network to evaluation mode

        if policy_file_path:
            self.load_policy(file_path=policy_file_path)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
    
    def load_policy(self, file_path):
        # Load pre-trained policy from a file
        self.policy_net.load_state_dict(torch.load(file_path, weights_only=False))
        self.policy_net.eval()  # Set the model to evaluation mode
        print(f"Model loaded from {file_path}")

    def select_action(self, observation, reward, termination, truncation, info, training=True):
        # Select an action based on the current state
        state = observation["observation"]
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_mask = torch.tensor(info.get("action_mask", None), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(state).squeeze(0)  # Get Q-values for the current state

            if action_mask is not None:
                q_values[action_mask == 0] = -float('inf')  # Mask invalid actions

        return torch.argmax(q_values).item()  # Return the action with the highest Q-value

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return  # Skip training if there are not enough samples in the buffer

        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Move data to the appropriate device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Calculate current Q-values for the taken actions
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Calculate the target Q-values using the target network
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values

        # Compute the loss and update the policy network
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        # Update the target network with the weights of the policy network
        self.target_net.load_state_dict(self.policy_net.state_dict())
