"""
Who needs friends when you have agents?
"""


import os
import math
import random
from copy import deepcopy
from collections import namedtuple, deque
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class UCBDQNAgent:
    def __init__(self,
                 env,
                 learning_rate=1e-4,
                 gamma=0.99,
                 tau=0.005,
                 ucb_c=2.0,
                 batch_size=128):
        self.env = env

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.ucb_c = ucb_c
        self.lr = learning_rate
        self.batch_size = batch_size

        self.device = torch.device("cpu")

        # Environment parameters
        self.board_size = env.board_size
        self.n_observations = self.board_size * self.board_size + 1  # +1 for pie rule
        self.n_actions = env.action_spaces["player_1"].n

        # Neural networks
        self.policy_net = DQNNetwork(
            self.n_observations, self.n_actions
        ).to(self.device)
        self.target_net = DQNNetwork(
            self.n_observations, self.n_actions
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and Replay buffer
        self.optimizer = optim.AdamW(self.policy_net.parameters(),
                                     lr=self.lr,
                                     amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.action_counts = np.zeros(self.n_actions)
        self.total_steps = 0

        self.episode_rewards = []

    def load_model(self):
        if not hasattr(self, 'model_loaded'):
            sparse_flag = self.env.sparse_flag
            if sparse_flag:
                self.lr = 1e-4
                self.ucb_c = 2.0
            else:
                self.lr = 1e-5
                self.ucb_c = 4.0
            
            model_path = "sparse_ucb_dqn_agent.pt" if sparse_flag else "dense_ucb_dqn_agent.pt"
            print(f"Loading model: {model_path}")
            self.policy_net.load_state_dict(torch.load(model_path, weights_only=True))
            self.policy_net.eval()
            self.model_loaded = True

    # def copy(self, player_id):
    #     new_agent = DQNAgent(self.env, player_id)
    #     new_agent.policy_net.load_state_dict(self.policy_net.state_dict())
    #     new_agent.target_net.load_state_dict(self.target_net.state_dict())
    #     new_agent.memory = deepcopy(self.memory)
    #     return new_agent

    def get_state(self, observation):
        """Returns the flattened board concatenated with the pie rule."""
        # todo: implement env.state and then convert that to a tensor

        board = torch.from_numpy(observation["observation"]).float()
        pie_rule = torch.tensor(observation["pie_rule_used"]).float()
        state = torch.cat([board.flatten(), pie_rule.unsqueeze(0)])
        state = state.unsqueeze(0)

        assert state.shape == (1, self.n_observations)
        # assert state.shape == (self.n_observations,)

        return state

    def get_action(self, action):
        return torch.tensor([action], dtype=torch.long, device=self.device)

    def get_reward(self, reward):
        return torch.tensor([reward], device=self.device)

    def select_action(self,
                      observation, reward, termination, truncation, info,
                      evaluating=False):
        agent_id = "player_1" if info["direction"] == 1 else "player_2"
        self.load_model()
        state = self.get_state(observation)

        mask = info["action_mask"]
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device)
        if evaluating:
            with torch.no_grad():
                state = self.get_state(observation)
                q_values = self.policy_net(state)
                mask_tensor = torch.tensor(mask,
                                           dtype=torch.bool,
                                           device=self.device).unsqueeze(0)
                q_values[~mask_tensor] = -float("inf")
                action = q_values.argmax().item()
                return action

        # UCB action selection
        with torch.no_grad():
            q_values = self.policy_net(state).squeeze()
        
        # Calculate UCB scores
        ucb_scores = np.zeros(self.n_actions)
        for action in range(self.n_actions):
            if mask[action]:
                if self.action_counts[action] > 0:
                    exploitation = q_values[action].item()
                    exploration = self.ucb_c * math.sqrt(
                        math.log(self.total_steps + 1) / self.action_counts[action]
                    )
                    ucb_scores[action] = exploitation + exploration
                else:
                    ucb_scores[action] = float('inf')  # Encourage exploring unvisited actions
            else:
                ucb_scores[action] = -float('inf')  # Invalid action
        
        action = np.argmax(ucb_scores)
        self.action_counts[action] += 1
        self.total_steps += 1
        
        return action

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return  # not enough samples to train

        # Sample a batch of experiences.
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))  # see readme to unpack this
        # brief: [(s, a, s, r), ...] -> [(s1, s2, ...), (a1, a2, ...), ...]

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)  # [batch_size, state_size]
        action_batch = torch.cat(batch.action)  # [batch_size]
        reward_batch = torch.cat(batch.reward)  # [batch_size]

        # Compute Q(s, a) for actions in the batch.
        state_action_values = self.policy_net(
            state_batch
        ).gather(1, action_batch.unsqueeze(1))  # [batch_size, 1]

        # Compute V(s) for all next states.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states
            ).max(1).values

        expected_state_action_values = ((next_state_values * self.gamma)
                                        + reward_batch)  # [batch_size, 1]

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self, agents, num_episodes=100, model_dir="./ucb_models_dense/"):
        """Train the agent."""

        os.makedirs(model_dir, exist_ok=True)

        env = self.env

        for i_episode in trange(1, num_episodes + 1, desc="Training", unit="episode"):
            env.reset()

            # Alternate roles every episode
            if i_episode % 2 == 0:
                current_agents = {
                    "player_1": agents["player_2"],
                    "player_2": agents["player_1"]
                }
            else:
                current_agents = agents

            episode_reward = 0
            done = False
            while not done:
                for agent in env.agent_iter():
                    observation, reward, termination, truncation, info = env.last()
                    done = termination or truncation

                    if done:
                        action = None
                    else:
                        action = current_agents[agent].select_action(
                            observation, reward, termination, truncation, info
                        )

                    if agent == "player_1":
                        episode_reward += reward

                    env.step(action)

                    # store the transition and train dqn_agent
                    if not done and agent == "player_1":
                        next_obs, reward, termination, truncation, info = env.last()
                        state = self.get_state(observation)
                        action = self.get_action(action)
                        next_state = self.get_state(next_obs)
                        reward = self.get_reward(reward)
                        self.memory.push(state, action, next_state, reward)
                        self.optimize_model()

                        # soft update target network
                        target_net_state_dict = self.target_net.state_dict()
                        policy_net_state_dict = self.policy_net.state_dict()
                        for key in policy_net_state_dict:
                            target_net_state_dict[key] = (policy_net_state_dict[key] * self.tau
                                                          + target_net_state_dict[key] * (1 - self.tau))
                        self.target_net.load_state_dict(target_net_state_dict)

            self.episode_rewards.append(episode_reward)

            if (i_episode) % 100 == 0:
                # print("Updating opponent agent...")
                # agents["player_2"] = agents["player_1"].copy("player_2")
                # agents["player_2"] = deepcopy(agents["player_1"])

                # Save the model
                filename = f"ucb_dqn_agent_e{i_episode}.pt"
                model_path = os.path.join(model_dir, filename)
                torch.save(self.policy_net.state_dict(), model_path)
                tqdm.write(f"Model saved to {model_path}")

    def plot_rewards(self):
        """Plot rewards over time."""

        smoothed_rewards = np.convolve(self.episode_rewards,
                                       np.ones(100) / 100,
                                       mode="valid")
        plt.plot(range(1, len(smoothed_rewards) + 1), smoothed_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Convergence: Rewards per Episode")
        plt.show()

    def evaluate(self, agents, num_episodes=10000, model_path="sparse_ucb_dqn_agent.pt"):
        """Evaluate the agent."""

        self.policy_net.load_state_dict(
            torch.load(model_path, weights_only=True))
            # torch.load("./best_dqn.pt", weights_only=True))
        self.policy_net.eval()

        env = self.env
        rewards = []

        for i_episode in trange(1, num_episodes + 1, desc="Evaluation", unit="episode"):
            env.reset()

            # episode_rewards = {agent: 0 for agent in agents.keys()}
            episode_reward = 0

            done = False
            while not done:
                for agent in env.agent_iter():
                    observation, reward, termination, truncation, info = env.last()
                    done = termination or truncation

                    # episode_rewards[agent] += reward
                    if agent == "player_1":
                        episode_reward = reward

                    if done:
                        action = None
                    else:
                        if agent == "player_1":
                            action = self.select_action(observation,
                                                        reward,
                                                        termination,
                                                        truncation,
                                                        info,
                                                        evaluating=True)
                        else:
                            action = agents[agent].select_action(
                                observation, reward, termination, truncation, info
                            )

                    env.step(action)

            # rewards.append(episode_rewards["player_1"])
            rewards.append(episode_reward)

        smoothed_rewards = np.convolve(
            rewards, np.ones(100) / 100, mode="valid")
        # plt.plot(range(1, len(smoothed_rewards) + 1), smoothed_rewards)
        # plt.xlabel("Episode")
        # plt.ylabel("Reward")
        # plt.title("Evaluation: Rewards per Episode")
        # plt.show()

        return rewards
