import numpy as np
from pettingzoo.utils.env import AECEnv
from gymnasium import spaces
from pettingzoo.utils import agent_selector


class OurHexGame(AECEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, board_size=5):
        super().__init__()
        self.board_size = board_size
        self.agents = ["player_1", "player_2"]
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.next()
        self.is_pie_rule_usable = False
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)

        self.action_space = spaces.Discrete(self.board_size * self.board_size + 1)
        self.action_spaces = {agent: self.action_space for agent in self.agents}

        self.observation_spaces = {
            agent: spaces.Box(
                low=0, high=2, shape=(self.board_size, self.board_size), dtype=np.int8
            )
            for agent in self.agents
        }

        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: [] for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.is_pie_rule_usable = False
        self.agent_selection = "player_1"
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: [] for agent in self.agents}

        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

    def step(self, action):
        # Handle pie rule
        if action == self.board_size * self.board_size:
            if self.agent_selection == "player_1":
                raise ValueError("Illegal move: Pie rule can only be used by Player 2.")
            if not self.is_pie_rule_usable:
                raise ValueError("Illegal move: Pie rule can only be used once.")

            self.is_pie_rule_usable = True

            self.board = np.where(self.board == 1, 3, self.board)
            self.board = np.where(self.board == 2, 1, self.board)
            self.board = np.where(self.board == 3, 2, self.board)
        else:
            row, col = divmod(action, self.board_size)

            # Ensure the chosen spot is empty
            if self.board[row, col] != 0:
                raise ValueError("Illegal move: Cell already occupied.")

            marker = 1 if self.agent_selection == "player_1" else 2
            self.board[row, col] = marker

            if self.check_winner(marker):
                self.terminations = {agent: True for agent in self.agents}
                self.rewards = {
                    agent: 10 if agent == self.agent_selection else -10
                    for agent in self.agents
                }
            else:
                self.rewards = {agent: -1 for agent in self.agents}
                self.terminations = {agent: False for agent in self.agents}

        if self.agent_selection == "player_2":
            # Pie rule should only be usable on the first move of Player 2
            self.is_pie_rule_usable = True

        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]

        self.agent_selection = self.agent_selector.next()

    def observe(self, agent):
        return self.board

    def render(self, mode="human"):
        pass

    def check_winner(self, marker):
        visited = set()
        if marker == 1:
            # Check DFS from all top row cells to see if Player 1 has reached the bottom
            for col in range(self.board_size):
                if self.board[0, col] == marker:
                    if self.dfs(marker, 0, col, visited):
                        return True
        else:
            # Check DFS from all top row cells to see if Player 1 has reached the bottom
            for row in range(self.board_size):
                if self.board[row, 0] == marker:
                    if self.dfs(marker, row, 0, visited):
                        return True
        return False

    def dfs(self, marker, row, col, visited):
        if marker == 1 and row == self.board_size - 1:  # Player 1 reached bottom
            return True
        if marker == 2 and col == self.board_size - 1:  # Player 2 reached right side
            return True

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
        visited.add((row, col))

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (
                0 <= new_row < self.board_size
                and 0 <= new_col < self.board_size
                and (new_row, new_col) not in visited
                and self.board[new_row, new_col] == marker
            ):
                if self.dfs(marker, new_row, new_col, visited):
                    return True

        return False

    def close(self):
        pass
