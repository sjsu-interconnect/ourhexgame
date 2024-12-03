import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class G12Agent(nn.Module):
    def __init__(self, env, gamma=0.99, lr=3e-4, epsilon=0.2, update_steps=10, l1=64, l2=128):
        super(G12Agent, self).__init__()
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.update_steps = update_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural network layers
        self.shared_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(env.board_size ** 2, l2),
            nn.ReLU(),
            nn.Linear(l2, l1),
            nn.ReLU(),
        ).to(self.device)

        self.policy_head = nn.Linear(l1, env.action_spaces[env.possible_agents[0]].n).to(self.device)
        self.value_head = nn.Linear(l1, 1).to(self.device)

        self.optimizer = optim.Adam(
            list(self.shared_layers.parameters()) +
            list(self.policy_head.parameters()) +
            list(self.value_head.parameters()),
            lr=self.lr
        )

        # Storage for transitions
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

    def select_action(self, observation, reward, termination, truncation, info):
        obs = observation["observation"].flatten()
        # Convert to PyTorch tensor
        state = torch.tensor(obs, dtype=torch.float32).to(self.device)
        state = state.unsqueeze(0)  # Add batch dimension

        # Forward pass
        with torch.no_grad():
            shared_output = self.shared_layers(state)
            logits = self.policy_head(shared_output)
            probs = Categorical(logits=logits)
        
        action = probs.sample()
        while not info["action_mask"][action.item()]:
            action = probs.sample()

        # Store log probability for PPO
        self.log_probs.append(probs.log_prob(action))
        return action.item()

    def store_transition(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def compute_returns(self):
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    def update(self):
        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.actions).to(self.device)
        returns = self.compute_returns()

        # Detach stored log probabilities
        old_log_probs = torch.cat(self.log_probs).detach()

        for _ in range(self.update_steps):
            shared_output = self.shared_layers(states)
            logits = self.policy_head(shared_output)
            values = self.value_head(shared_output).squeeze()

            probs = Categorical(logits=logits)
            log_probs = probs.log_prob(actions)

            # Compute advantages
            advantages = returns - values
            ratio = torch.exp(log_probs - old_log_probs)

            # Compute surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values, returns)
            loss = policy_loss + 0.5 * value_loss

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear the transition buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
    
# Save the model
def save_model(model, path='g12agent.pth'):
    torch.save(model.state_dict(), path)

# Load the model
def load_model(model, path='g12agent.pth'):
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()