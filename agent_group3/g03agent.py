import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from ourhexenv import OurHexGame


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class G03Agent:
    def __init__(self, env, player_id="player_1", gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, learning_rate=1e-3, buffer_size=10000, c=1.0):
        """
        Initialize the agent for the given environment.

        Args:
            env: The OurHexGame environment instance.
        """
        self.player_id = player_id  # Player identifier
        self.state_size = env.observation_spaces[player_id]["observation"].shape[0] * env.observation_spaces[player_id]["observation"].shape[1]
        self.action_size = env.action_spaces[player_id].n

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.action_counts = np.zeros(self.action_size)  
        self.total_action_count = 0 

        self.c = c


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-network
        self.q_network = QNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network = QNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Replay buffer
        self.memory = deque(maxlen=buffer_size)

        # Synchronize target network
        self.update_target_network()

    def update_target_network(self):
        """Copy weights from the main Q-network to the target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def select_action(self, observation, reward, termination, truncation, info):
        """
        Select an action based on the current observation.
        """
        state = observation["observation"].flatten()
        action_mask = info["action_mask"]  # Access action_mask

        # Check if Pie Rule is valid
        is_pie_rule_valid = (
            self.player_id == "player_2"  # Player 2 only
            and observation.get("pie_rule_used", 0) == 0  # Pie Rule not yet used
            and action_mask[-1] == 1  # Pie Rule action is valid
        )

        # Player 2's first turn: Select Pie Rule if valid
        if is_pie_rule_valid and random.random() < self.epsilon:
            print("Player 2 is selecting the Pie Rule.")
            return len(action_mask) - 1  # Last action corresponds to Pie Rule

        # Regular epsilon-greedy action selection
        '''
        if random.random() < self.epsilon:
            # Choose a random valid action
            valid_actions = np.where(action_mask == 1)[0]
            return int(random.choice(valid_actions))
        '''
        if random.random() < self.epsilon:
            # Using UCB
            self.total_action_count += 1
            state_tensor = torch.FloatTensor(observation["observation"].flatten()).unsqueeze(0).to(self.device)

            # Predict Q-values for all actions
            with torch.no_grad():
                q_values = self.q_network(state_tensor).cpu().numpy()[0]

            # Mask invalid actions by setting their Q-values to a very low value
            q_values = np.where(action_mask, q_values, -np.inf)

            # Apply UCB formula to modify Q-values
            ucb_values = q_values + self.c * np.sqrt(np.log(self.total_action_count + 1) / (self.action_counts + 1e-5))
            valid_ucb_values = np.where(action_mask, ucb_values, -np.inf)

            # Select the best valid action based on UCB
            best_ucb_action = int(np.argmax(valid_ucb_values))
            
            # Update action counts
            self.action_counts[best_ucb_action] += 1

            return best_ucb_action

        # Predict Q-values for all actions
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]

        # Mask invalid actions by setting their Q-values to a very low value
        q_values = np.where(action_mask, q_values, -np.inf)

        # Choose the best valid action
        return int(np.argmax(q_values))

    def train(self, batch_size=64):
        """Train the Q-network using experiences from the replay buffer."""
        if len(self.memory) < batch_size:
            print(f"Not enough experiences in memory: {len(self.memory)}/{batch_size}")
            return

        # Sample a minibatch of experiences
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        # Convert data to tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # Compute current Q-values and target Q-values
        q_values = self.q_network(state).gather(1, action).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_state).max(1)[0]
            targets = reward + (1 - done) * self.gamma * next_q_values

        # Compute loss and update the Q-network
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon for exploration
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Log training progress
        print(f"Training step: Loss={loss.item():.4f}, Epsilon={self.epsilon:.4f}")


    def train_agent(num_episodes=500, board_size=11, sparse_flag=True):
        env = OurHexGame(board_size=board_size, sparse_flag=True)
        env.reset()

        agent = G03Agent(env)
        def modify_dense_reward(observation, reward, agent_id):
            """
            Give reward at each stage based on the player's actions.
            """
            # Get the board state
            board = observation["observation"]

            # Determine the current player's marker
            player_marker = 1 if agent_id == "player_1" else 2
            opponent_marker = 2 if agent_id == "player_1" else 1

            # Center position
            center = np.array([board.shape[0] // 2, board.shape[1] // 2])

            # Get positions filled by the current player
            filled_positions = np.argwhere(board == player_marker)

            # Reward for central positioning
            if filled_positions.size > 0:
                distances_to_center = np.linalg.norm(filled_positions - center, axis=1)
                center_reward = -np.mean(distances_to_center)  # Closer to center gets higher reward
            else:
                center_reward = 0

            # Reward for blocking opponent
            blocking_reward = 0
            opponent_positions = np.argwhere(board == opponent_marker)
            for pos in opponent_positions:
                neighbors = [
                    (pos[0] + dr, pos[1] + dc)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
                ]
                for neighbor in neighbors:
                    if (
                        0 <= neighbor[0] < board.shape[0]
                        and 0 <= neighbor[1] < board.shape[1]
                        and board[neighbor] == player_marker
                    ):
                        blocking_reward += 0.1  # Small reward for blocking opponent's path

            # Reward for expanding territory
            expansion_reward = len(filled_positions) * 0.05  # Proportional to the number of tiles occupied

            # Combine the rewards
            total_reward = reward + 0.1 * center_reward + blocking_reward + expansion_reward
            return total_reward


        for episode in range(num_episodes):
            env.reset()
            done = False
            while not done:
                for agent_id in env.agent_iter():
                    observation, reward, termination, truncation, info = env.last()

                    if termination or truncation:
                        done = True
                        break

                    # Modify reward if using Dense Reward
                    if not sparse_flag:
                        reward = modify_dense_reward(observation, reward, agent_id)

                    action = agent.select_action(observation, reward, termination, truncation, info)

                    next_observation = env.observe(agent_id)["observation"].flatten()
                    agent.store_transition(
                        observation["observation"].flatten(),
                        action,
                        reward,
                        next_observation,
                        termination,
                    )
                    agent.train(batch_size=64)

                    env.step(action)

            print(f"Episode {episode + 1}/{num_episodes} complete. Epsilon: {agent.epsilon:.4f}")

        model_name = "trained_sparse_agent.pth" if sparse_flag else "trained_dense_agent.pth"
        agent.save_model(model_name)
        print(f"Training complete. Model saved as '{model_name}'.")




    def save_model(self, file_name="gxx_agent.pth"):
        """Save the Q-network model."""
        torch.save(self.q_network.state_dict(), file_name)

    def load_model(self, file_name="gxx_agent.pth"):
        """Load the Q-network model."""
        self.q_network.load_state_dict(torch.load(file_name))
        self.update_target_network()
