import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class PPOActorCritic(nn.Module):
    """Combined Actor-Critic Network for PPO."""
    def __init__(self, state_size, action_size):
        super(PPOActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(256, action_size),
            nn.Softmax(dim=-1)  # Outputs probabilities for actions
        )
        self.critic = nn.Linear(256, 1)  # Outputs state-value

    def forward(self, state):
        shared_output = self.shared(state)
        policy = self.actor(shared_output)
        value = self.critic(shared_output)
        return policy, value


class GYYAgent:
    def __init__(self, env, gamma=0.99, actor_lr=0.0003, critic_lr=0.0003, batch_size=64,
                 memory_size=20000, ppo_epochs=4, clip_epsilon=0.2, entropy_coef=0.01):
        self.env = env
        self.state_size = np.prod(env.board.shape)
        self.action_size = env.board_size * env.board_size + 1  # Including the pie rule
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.ppo_epochs = ppo_epochs
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef

        # Replay buffer
        self.memory = deque(maxlen=self.memory_size)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.actor_critic = PPOActorCritic(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.actor_lr)

    def remember(self, state, action, reward, next_state, done):
        """Store experiences in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, observation, reward, termination, truncation, info):
        """Select an action using the policy."""
        state = observation["observation"].flatten()
        valid_actions = [
            action for action in range(self.action_size - 1) if state[action] == 0
        ]
        if info.get("pie_rule_allowed", False):
            valid_actions.append(self.action_size - 1)

        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            policy, _ = self.actor_critic(state_tensor)
        policy = policy.cpu().numpy().flatten()

        valid_action_probs = [policy[a] for a in valid_actions]
        valid_action_probs /= np.sum(valid_action_probs)
        return np.random.choice(valid_actions, p=valid_action_probs)

    def train(self):
        """Train the PPO model."""
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

        # Compute returns and advantages
        with torch.no_grad():
            _, values = self.actor_critic(states)
            values = values.squeeze()
            _, next_values = self.actor_critic(next_states)
            next_values = next_values.squeeze()
            returns = rewards + self.gamma * (1 - dones) * next_values
            advantages = returns - values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert actions to one-hot encoding for policy update
        actions_one_hot = torch.zeros(len(actions), self.action_size).to(self.device)
        actions_one_hot.scatter_(1, actions.unsqueeze(1), 1)

        old_policies = torch.zeros(len(actions), self.action_size).to(self.device)
        for state, action in zip(states, actions):
            old_policy, _ = self.actor_critic(state.unsqueeze(0))
            old_policies[state] = old_policy

        # PPO updates
        for _ in range(self.ppo_epochs):
            # Compute current policy and value
            policies, values = self.actor_critic(states)
            policy_probs = torch.sum(policies * actions_one_hot, dim=1)
            old_policy_probs = torch.sum(old_policies * actions_one_hot, dim=1)
            ratios = policy_probs / (old_policy_probs + 1e-8)

            # Compute clipped objective
            clipped_ratios = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            actor_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

            # Critic loss
            value_loss = nn.MSELoss()(values.squeeze(), returns)

            # Entropy loss for better exploration
            entropy_loss = -torch.mean(torch.sum(policies * torch.log(policies + 1e-8), dim=1))

            # Combined loss
            loss = actor_loss + value_loss + self.entropy_coef * entropy_loss

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
