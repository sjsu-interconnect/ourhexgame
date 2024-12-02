from ourhexenv import OurHexGame
import random
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from math import sqrt, log


# gamestate.py
import numpy as np

class GameState:
    def __init__(self, board, current_player, pie_rule_used, winner=None):
        self.board = board.copy()
        self.current_player = current_player
        self.pie_rule_used = pie_rule_used
        self.winner = winner

    def get_valid_actions(self, board_size):
        valid = (self.board == 0).astype(np.float32).flatten()
        # Handle pie rule (action index = board_size * board_size)
        pie_rule_valid = 1.0 if (not self.pie_rule_used and
                                 self.current_player == "player_2") else 0.0
        valid = np.append(valid, pie_rule_valid)
        return [i for i, v in enumerate(valid) if v > 0]

    def apply_action(self, action, board_size):
        new_board = self.board.copy()
        pie_rule_used = self.pie_rule_used
        winner = self.winner

        if action == board_size * board_size:
            # Pie rule action
            if self.current_player == "player_2" and not self.pie_rule_used:
                pie_rule_used = True
                # Swap stones for pie rule
                new_board[new_board == 1] = 3
                new_board[new_board == 2] = 1
                new_board[new_board == 3] = 2
        else:
            x, y = divmod(action, board_size)
            if new_board[x, y] == 0:
                new_board[x, y] = 1 if self.current_player == "player_1" else 2
                # Check for winner here and set winner variable if the game has ended
                winner = self.check_winner(new_board, board_size)
            else:
                # Invalid action; should not happen if MCTS is working correctly
                pass

        next_player = "player_2" if self.current_player == "player_1" else "player_1"
        return GameState(new_board, next_player, pie_rule_used, winner)

    def is_terminal(self):
        return self.winner is not None

    def check_winner(self, board, board_size):
        """
        Check if there's a winner in the current board state.

        For Hex, a player wins by forming a connected path from one side to the opposite side.
        """
        def dfs(x, y, visited, player):
            if (x, y) in visited:
                return False
            visited.add((x, y))
            if player == 1 and y == board_size - 1:
                return True
            if player == 2 and x == board_size - 1:
                return True
            directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < board_size and 0 <= ny < board_size and board[nx, ny] == player:
                    if dfs(nx, ny, visited, player):
                        return True
            return False

        for player in [1, 2]:
            visited = set()
            if player == 1:
                # Player 1 connects top to bottom (y=0 to y=board_size-1)
                for x in range(board_size):
                    if board[x, 0] == player:
                        if dfs(x, 0, visited, player):
                            return "player_1"
            else:
                # Player 2 connects left to right (x=0 to x=board_size-1)
                for y in range(board_size):
                    if board[0, y] == player:
                        if dfs(0, y, visited, player):
                            return "player_2"
        return None  # No winner yet

    
# -------------------------------
# PolicyValueNetwork for MCTSAgent (outputs policy and value)
# -------------------------------
class PolicyValueNetwork(nn.Module):
    def __init__(self, board_size, num_actions):
        super(PolicyValueNetwork, self).__init__()
        self.board_size = board_size
        self.num_actions = num_actions

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Policy head
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, num_actions)

        # Value head
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Common layers
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))

        # Policy head
        p = torch.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)
        policy = torch.softmax(policy_logits, dim=1)

        # Value head
        v = torch.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value
    
class Node:
    def __init__(self, state, parent=None, prior_p=1.0, action=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior_p = prior_p
        self.action = action

    def expand(self, action_probs, board_size):
        for action, prob in action_probs.items():
            next_state = self.state.apply_action(action, board_size)
            self.children[action] = Node(next_state, parent=self, prior_p=prob, action=action)

    def is_leaf(self):
        return len(self.children) == 0

    def select_child(self, c_puct):
        best_value = -float('inf')
        best_node = None
        for child in self.children.values():
            ucb_value = (child.total_value / (child.visit_count + 1e-8) +
                         c_puct * child.prior_p * sqrt(self.visit_count) / (1 + child.visit_count))
            if ucb_value > best_value:
                best_value = ucb_value
                best_node = child
        return best_node

    def backpropagate(self, value):
        self.visit_count += 1
        self.total_value += value
        if self.parent:
            self.parent.backpropagate(-value)  # Switch perspective

policyValueNetwork = PolicyValueNetwork(board_size=11, num_actions=11*11+1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

class MCTSAgent:
    def __init__(self, env, neural_network=None, num_simulations=800, c_puct=1.0, dirichlet_alpha=0.03, epsilon=0.25):
        self.env = env
        if neural_network is None:
            self.neural_network = PolicyValueNetwork(env.board_size, env.board_size * env.board_size + 1)
            # Optionally load pretrained weights here
        else:
            self.neural_network = neural_network
        self.neural_network.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.neural_network.eval()  # Set to evaluation mode
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.board_size = env.board_size
        self.num_actions = self.board_size * self.board_size + 1
        self.dirichlet_alpha = dirichlet_alpha
        self.epsilon = epsilon

    def select_action(self, observation, reward, termination, truncation, info):
        """
        Select an action based on the current observation and other parameters.

        Parameters:
            observation (dict): The current observation from the environment.
            reward (float): The reward received from the previous action.
            termination (bool): Whether the episode has terminated.
            truncation (bool): Whether the episode was truncated.
            info (dict): Additional information from the environment.

        Returns:
            int: The selected action.
        """
        if termination or truncation:
            self.handle_episode_end(reward, info)

        root_state = self.create_game_state(observation)
        root_node = Node(state=root_state)

        for _ in range(self.num_simulations):
            node = root_node
            # Selection
            while not node.is_leaf():
                node = node.select_child(self.c_puct)

            # Expansion
            if not node.state.is_terminal():
                self.expand_node(node)

            # Simulation
            value = self.simulate(node.state)

            # Backpropagation
            node.backpropagate(value)

        # Choose the action with the highest visit count
        if not root_node.children:
            # If no children were expanded, choose a random valid action
            valid_actions = root_state.get_valid_actions(self.board_size)
            best_action = random.choice(valid_actions)
        else:
            best_action = max(root_node.children.items(), key=lambda item: item[1].visit_count)[0]
        return best_action

    def handle_episode_end(self, reward, info):
        """
        Handle the end of an episode. This can be used to reset internal states or perform logging.

        Parameters:
            reward (float): The reward received from the last action.
            info (dict): Additional information from the environment.
        """
        # Implement any necessary logic when an episode ends
        # For example, you might reset internal data structures or log the outcome
        pass

    def create_game_state(self, observation):
        board = observation["observation"]
        pie_rule_used = observation.get("pie_rule_used", False)
        current_player = observation.get("current_player", "player_1")  # Adjust based on your observation structure
        return GameState(board, current_player, pie_rule_used)

    def expand_node(self, node):
        state = node.state
        valid_actions = state.get_valid_actions(self.board_size)
        state_tensor = self.preprocess_state(state)
        with torch.no_grad():
            policy_logits, value = self.neural_network(state_tensor)
        policy = policy_logits.cpu().numpy().flatten()
        policy = np.exp(policy)
        policy = policy / np.sum(policy)
        action_probs = {a: p for a, p in enumerate(policy) if a in valid_actions}
        node.expand(action_probs, self.board_size)

    def simulate(self, state):
        # Use the value predicted by the neural network
        state_tensor = self.preprocess_state(state)
        with torch.no_grad():
            _, value = self.neural_network(state_tensor)
        return value.item()

    def preprocess_state(self, state):
        board = state.board
        current_player = state.current_player
        pie_rule_used = state.pie_rule_used

        player1 = (board == 1).astype(np.float32)
        player2 = (board == 2).astype(np.float32)
        current_player_channel = np.full(board.shape, 1.0 if current_player == "player_1" else 0.0, dtype=np.float32)
        pie_rule_channel = np.full(board.shape, float(pie_rule_used), dtype=np.float32)

        state_array = np.stack([player1, player2, current_player_channel, pie_rule_channel], axis=0)
        state_tensor = torch.from_numpy(state_array).unsqueeze(0).to(self.device)
        return state_tensor
    
def load_mcts_agent(env, filepath):
    neural_network = PolicyValueNetwork(env.board_size, env.board_size * env.board_size + 1)
    neural_network.load_state_dict(torch.load(filepath, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    neural_network.eval()  # Set to evaluation mode
    agent = MCTSAgent(env, neural_network)
    return agent