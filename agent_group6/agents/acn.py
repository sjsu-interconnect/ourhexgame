import os
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class ActorCriticNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, actor_lr, critic_lr, fc1_dims=512, fc2_dims=512, fc3_dims=256, fc4_dims=256):
        super(ActorCriticNetwork, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, fc3_dims),
            nn.ReLU(),
            nn.Linear(fc3_dims, fc4_dims),
            nn.ReLU(),
            nn.Linear(fc4_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, fc3_dims),
            nn.ReLU(),
            nn.Linear(fc3_dims, fc4_dims),
            nn.ReLU(),
            nn.Linear(fc4_dims, 1)
        )

        self.actor_optimizer = optim.Adam(self.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.parameters(), lr=critic_lr)
        self.device = "cuda:0" if T.cuda.is_available() else "cpu"

    def forward(self, state):
        dist = self.actor(state)
        value = self.critic(state)
        return dist, value
    
    def print_info(self, file=None):
        if file:
            with open(file, 'a') as f:
                f.write("Actor-Critic Network Architecture:\n")
                f.write("Actor:\n")
                f.write(str(self.actor) + "\n")
                f.write("\nCritic:\n")
                f.write(str(self.critic) + "\n")
                f.write("\nDevice: " + str(self.device) + "\n")
                f.write("\nParameters:\n")
                for name, param in self.named_parameters():
                    f.write(f"{name}: {param.shape}\n")
    
    def act(self, state: np.ndarray, mask: np.ndarray):
        """Compute action and log probabilities."""
        probs = self.actor(state)
        # Convert probs to numpy array
        probs_np = probs.detach().cpu().numpy().flatten()

        # Apply mask
        valid_actions = mask.astype(bool)
        probs_np[~valid_actions] = 0

        # Normalize probabilities
        if np.sum(probs_np) > 0:
            probs_np /= np.sum(probs_np)
        else:
            probs_np[valid_actions] = 1.0 / np.sum(valid_actions)
            # import ipdb; ipdb.set_trace()

        # Choose action
        action = np.random.choice(len(probs_np), p=probs_np)

        # Convert back to tensor
        action_tensor = T.tensor([action], dtype=T.long).to(self.device)
        probs_tensor = T.tensor([probs_np[action]], dtype=T.float).to(self.device)

        # Compute log probability and state value
        action_log_prob = T.log(probs_tensor)
        state_val = self.critic(state)
        return action_tensor.detach(), action_log_prob.detach(), state_val.detach()
    
    def evaluate(self, state: np.ndarray, action: T.Tensor):
        """Evaluate state for critic value."""
        probs = self.actor(state)
        distribution = T.distributions.Categorical(probs)
        action_log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        state_val = self.critic(state)
        return state_val, action_log_prob, entropy
