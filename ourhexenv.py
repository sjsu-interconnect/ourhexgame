import math
import pygame
import numpy as np
from pettingzoo.utils.env import AECEnv
from gymnasium import spaces
from pettingzoo.utils import agent_selector
from UnionFind import UnionFind


class OurHexGame(AECEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, board_size=5, sparse_flag=True):
        super().__init__()
        self.board_size = board_size
        self.sparse_flag = sparse_flag
        self.agents = ["player_1", "player_2"]
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.next()
        self.is_pie_rule_usable = False
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)

        self.action_space = spaces.Discrete(self.board_size * self.board_size + 1)
        self.action_spaces = {agent: self.action_space for agent in self.agents}

        self.observation_spaces = {
            agent: spaces.Dict({
                "observation": spaces.Box(low=0,
                                          high=2,
                                          shape=(self.board_size, self.board_size),
                                          dtype=np.int8),
                "pie_rule_used": spaces.Discrete(2), # 1 if used, 0 otherwise
            })
            for agent in self.agents
        }

        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: [] for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}

        # Pygame setup
        self.window = None
        self.clock = None
        self.cell_size = 30
        self.hex_radius = self.cell_size // 2
        self.width = int(self.cell_size * (board_size * 2.25))
        self.height = int(self.cell_size * (board_size * 1.25)) + 10
        self.hex_points_cache = {}
        
        # Colors
        self.BACKGROUND = (200, 200, 200)
        self.GRID = (100, 100, 100)
        self.PLAYER1 = (255, 50, 50)    # Red
        self.PLAYER2 = (50, 50, 255)    # Blue
        self.EMPTY = (255, 255, 255)    # White

        # Union Find Check Winner Setup
        self.uf = UnionFind(board_size * board_size + 4)  # Extra 4 for virtual nodes
        self.top_virtual = board_size * board_size        # player_1 owns top + bottom nodes
        self.bottom_virtual = self.top_virtual + 1
        self.left_virtual = self.top_virtual + 2          # player_2 owns left + right nodes
        self.right_virtual = self.top_virtual + 3

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        
        self.is_first = True
        self.is_pie_rule_usable = False
        self.agent_selection = "player_1"

        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: [] for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}

        
        if self.window:
            self.window.fill(self.BACKGROUND)
            pygame.display.flip()
        

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
            self.place_piece(row, col, marker)

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


    def place_piece(self, row, col, marker):
        """
        Record the player's turn by marking their selected tile.
        Maintain the 'Union Find Check Winner Stucture'.
        """
        pos = row * self.board_size + col

        # Connect to adjacent cells
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == marker:
                self.uf.union(pos, r * self.board_size + c)

        # Connect to virtual nodes if on border
        if marker == 1:  # First player connects top-bottom
            if row == 0:
                self.uf.union(pos, self.top_virtual)
            if row == self.board_size - 1:
                self.uf.union(pos, self.bottom_virtual)
        elif marker == 2:  # Second player connects left-right
            if col == 0:
                self.uf.union(pos, self.left_virtual)
            if col == self.board_size - 1:
                self.uf.union(pos, self.right_virtual)

        self.board[row, col] = marker


    def check_winner(self, player):
        """
        Check whether a certain player has won the game
        Verify whether virtual nodes on opposite sides of the board now belong to the same set.
        """
        if player == 1:  # First player
            return self.uf.find(self.top_virtual) == self.uf.find(self.bottom_virtual)
        elif player == 2:  # Second player
            return self.uf.find(self.left_virtual) == self.uf.find(self.right_virtual)
        return False


    def observe(self, agent):
        return self.board


    def _get_hex_points(self, x, y):
        if (x, y) in self.hex_points_cache:
            return self.hex_points_cache[(x, y)]
        points = []
        for i in range(6):
            angle_deg = 60 * i - 30
            angle_rad = math.pi / 180 * angle_deg
            point_x = x + self.hex_radius * math.cos(angle_rad)
            point_y = y + self.hex_radius * math.sin(angle_rad)
            points.append((point_x, point_y))
        self.hex_points_cache[(x, y)] = points
        return points

    def render(self, mode="human"):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height), pygame.SRCALPHA)
            pygame.display.set_caption("Hex Game")
            self.clock = pygame.time.Clock()
            self.hex_points_cache = {}

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            
        self.window.fill(self.BACKGROUND)

        # Draw the board
        for row in range(self.board_size):
            for col in range(self.board_size):
                x = self.cell_size * (1.4 * col + 1) + (row * self.cell_size * 0.75)
                y = self.cell_size * (row * math.sqrt(3)/2 + 1)
                
                points = self._get_hex_points(x, y)
                
                color = self.EMPTY
                if self.board[row, col] == 1:
                    color = self.PLAYER1
                elif self.board[row, col] == 2:
                    color = self.PLAYER2

                pygame.draw.polygon(self.window, color, points)
                pygame.draw.aalines(self.window, self.GRID, True, points, 2)
                
        # Player_1 borders (top-bottom)
        pygame.draw.line(self.window, self.PLAYER1, (self.cell_size, 0), 
                        (self.width - self.cell_size, 0), 5)
        pygame.draw.line(self.window, self.PLAYER1, (self.cell_size, self.height), 
                        (self.width - self.cell_size, self.height), 5)
                        
        # Player_2 borders (left-right)
        pygame.draw.line(self.window, self.PLAYER2, (0, self.cell_size), 
                        (0, self.height - self.cell_size), 5)
        pygame.draw.line(self.window, self.PLAYER2, (self.width, self.cell_size), 
                        (self.width, self.height - self.cell_size), 5)

        pygame.display.flip()
        self.clock.tick(30)


    # def check_winner(self, marker):
    #     visited = set()
    #     if marker == 1:
    #         # Check DFS from all top row cells to see if Player 1 has reached the bottom
    #         for col in range(self.board_size):
    #             if self.board[0, col] == marker:
    #                 if self.dfs(marker, 0, col, visited):
    #                     return True
    #     else:
    #         # Check DFS from all top row cells to see if Player 2 has reached the bottom
    #         for row in range(self.board_size):
    #             if self.board[row, 0] == marker:
    #                 if self.dfs(marker, row, 0, visited):
    #                     return True
    #     return False
    #
    # def dfs(self, marker, row, col, visited):
    #     if marker == 1 and row == self.board_size - 1:  # Player 1 reached bottom
    #         return True
    #     if marker == 2 and col == self.board_size - 1:  # Player 2 reached right side
    #         return True
    #
    #     directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
    #     visited.add((row, col))
    #
    #     for dr, dc in directions:
    #         new_row, new_col = row + dr, col + dc
    #         if (
    #             0 <= new_row < self.board_size
    #             and 0 <= new_col < self.board_size
    #             and (new_row, new_col) not in visited
    #             and self.board[new_row, new_col] == marker
    #         ):
    #             if self.dfs(marker, new_row, new_col, visited):
    #                 return True
    #
    #     return False

    def close(self):
        print("Called close")
        if self.window is not None:
            self.window = None
            self.clock = None
