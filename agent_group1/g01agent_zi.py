import copy
import random
import math
import warnings
import functools
import pygame
import json
import numpy as np
from pathlib import Path
from typing import Dict

from pettingzoo.utils.env import AECEnv
from gymnasium import spaces
from pettingzoo.utils import agent_selector

from ourhexenv import OurHexGame

# Agent class that contains 2 agents, a dumb pick a spot at random uniform agent,
# and a Q_learning based agent where it loads Q_values from previous runs and
# use that as values in the Q_value table to guide the Q_learning agent.
# The Q_learning agent, contained in the function select_action_basic_agent
# is the basic epsilon-decay agent since it uses a epsilon-greedy policy for exploration
class G01Agent_QL():
    def __init__(self, game_env, agent_selector=1, sparse_flag=True):
        super().__init__()
        self._env = game_env
        self._board_size = 11 * 11                   # board size stored in class
        self._Q_value_table = dict()                 # Q_value table as one giant dictionary
        self._epsilon = 0.1                          # epsilon for epsilon-greedy decay
        self._agent_selector = agent_selector        # selects using dumb or Q_learning agent 0 = dumb agent and 1 = Q_learning agent

        self._alpha = 0.02                           # Q_learning alpha
        self._gamma = 0.98                           # Q_learning gamma
        self._former_reward_R = 0                    # last reward value learned for Q_learning
        self._former_pie_rule_state_P = 0            # Pie rule used or not 0 = not used 1 = used
        self._former_board_state_S = None            # Former board state S for Q_learning
        self._former_action_A = None                 # Former action taken by agent A for Q_learning
        self._initial_state = True                   # agent at initial state before taking first action

        self._actions_taken = 0                      # number of actions taken so far in a game by the agent

        self._env_sparse_rewards = sparse_flag       # game environment

    # Action selection function that choose the agent based on agent selector and returns chosen action
    def select_action(self, env_observation, env_reward, env_termination, env_truncation, env_info):
        if self._agent_selector == 0:
            action_chosen = self.select_action_dumb_agent(env_observation, env_reward, env_termination, env_truncation, env_info)
        else:
            action_chosen = self.select_action_basic_agent(env_observation, env_reward, env_termination, env_truncation, env_info)

        return action_chosen

    # function to store epsilon
    def set_epsilon(self, epsilon: float):
        self._epsilon = epsilon
        pass

    # Function to update Q_values
    # Since there are 3^121 possible states, not possible to create space to store all possible Q_values
    # so Q_values with states never visited will always be zero unless that state has been visited by the game
    # at which point, an entry in the dictionary with a 121-character length string as a key is created with
    # the chosen action as key to the values dictionary to store the Q_value associated with State-Action pair
    def update_Q_value_table(self, S, P, A, value):
        Q_key = self.convert_board_array_to_key(S, P)
        if Q_key in self._Q_value_table.keys():
            self._Q_value_table[Q_key][A] = value
        else:
            self._Q_value_table[Q_key] = {A: float(value)}
        pass

    # Function to read Q_value from table or 0 if state never been visited
    def read_Q_value_table(self, S, P, A) -> float:
        Q_key = self.convert_board_array_to_key(S, P)
        if Q_key in self._Q_value_table.keys():
            if A in self._Q_value_table[Q_key].keys():
                ret_val = self._Q_value_table[Q_key][A]
            else:
                ret_val = 0.0
        else:
            ret_val = 0.0
        return ret_val

    # function to convert the observation board state into a 121-byte character string
    def convert_board_array_to_key(self, S, P: int) -> str:
        key_str = ""
        i = 0
        while i < len(S):
            j = 0
            while j < len(S[i]):
                key_str += str(S[i][j])
                j += 1
            i += 1
        key_str += str(P)
        return key_str

    # Get all possible actions in current state from observation information
    def get_all_possible_actions(self, S, P: int) -> list[int]:
        possible_actions_list = []
        action_value = 0
        i = 0
        while i < len(S):
            j = 0
            while j < len(S[i]):
                if S[i][j] == 0:
                    possible_actions_list.append(action_value)
                j += 1
                action_value += 1
            i += 1
        if (P == 0) and (self._actions_taken == 0):     # check to see if pie-rule is a valid action and if so include it
            possible_actions_list.append(action_value)
        return possible_actions_list

    # Function to get all the Q_values associated with each possible action in the current state
    # this function is needed to perform max_a_Q(S', a) for the Q_learning agent
    def get_all_Q_values_of_actions(self, S: list[list[int]], P: int,  AP_list: list[int]) -> list[float]:
        Q_values_list = []
        for a in AP_list:
            Q_values_list.append(self.read_Q_value_table(S, P, a))
        return Q_values_list

    # Function to compute and choose an action based on the current game state by using the epislon-greedy algorithm
    # so that a small possibility of exploration is built-in
    def get_epsilon_greedy_action(self, S, P: int) -> int:
        # get all possible actions and Q_values of all those actions
        possible_action_list = self.get_all_possible_actions(S, P)
        possible_action_Q_values_list = self.get_all_Q_values_of_actions(S, P,  possible_action_list)

        # Get maximum possible Q_value among all possible actions
        max_Q_value = max(possible_action_Q_values_list)
        max_Q_value_actions = list()
        i = 0
        for x in possible_action_Q_values_list:
            if x == max_Q_value:
                max_Q_value_actions.append(possible_action_list[i])
            i += 1

        # Choose an action based on greedy probability of 1-epsilon for max Q_value action
        # and epislon / #_possible_actions for all actions
        num_max_Q_value_actions = len(max_Q_value_actions)
        num_total_actions = len(possible_action_list)
        unit_rand = random.uniform(0, 1)
        if unit_rand > 1-self._epsilon:
            action_idx = random.randint(0, num_total_actions-1)
            action_chosen = possible_action_list[action_idx]
        else:
            action_idx = random.randint(0, num_max_Q_value_actions-1)
            action_chosen = max_Q_value_actions[action_idx]

        return(action_chosen)

    # Function that represents the dumb agent that chooses an action uniformly randomly among all possible actions
    # at every step of the game
    def select_action_dumb_agent(self, env_observation, env_reward, env_termination, env_truncation, env_info) -> int:
        number_of_valid_moves = np.count_nonzero(env_info["action_mask"] == 1)
        if number_of_valid_moves > 0:
            move_choice = random.randint(0, self._board_size)
            empty_space_not_found = True
            while empty_space_not_found:
                if env_info["action_mask"][move_choice] == 1:
                    empty_space_not_found = False
                else:
                    move_choice = random.randint(0, self._board_size)
        else:
            move_choice = None
        self._actions_taken += 1
        return move_choice

    # Function that gets the state of the board from the environment passed observation parameter
    def get_board_state_from_env_obs(self, env_observation):
        board_state_array = env_observation["observation"]
        return (copy.deepcopy(board_state_array))

    # Function that gets the state of whether the pie-rule is used from the environment passed observation parameter
    def get_pie_rule_state_from_env_obs(self, env_observation) -> int:
        ret_val = env_observation["pie_rule_used"]
        return ret_val

    # Function to get the action that has the highest possible Q_value from the game's current board state
    def max_Q_action(self, S, P: int) -> int:
        possible_action_list = self.get_all_possible_actions(S, P)
        max_Q_value = -130
        max_a = 0
        for a in possible_action_list:
            current_Q_value = self.read_Q_value_table(S, P, a)
            if current_Q_value > max_Q_value:
                max_Q_value = current_Q_value
                max_a = a
        return max_a

    # Function that returns the decides what the next action will be for the Q_learning agent.
    # Here is the main steps of the Q_learning agent where the Q_values are updated from max_a_Q and old Q_values
    def select_action_basic_agent(self, env_observation, env_reward, env_termination, env_truncation, env_info) -> int:
        # If first state of game then no need to update Q_value, just save state as S, and state of pie-rule
        # also load Q_values obtained from past game plays
        if self._initial_state == True:
            self._former_board_state_S = self.get_board_state_from_env_obs(env_observation)
            self._former_pie_rule_state_P = self.get_pie_rule_state_from_env_obs(env_observation)
            self._initial_state = False
            self.load_past_experience()
        else:   # if not first state then update the Q_value based on the Q_Learning algorithm and update S <- S'
            self._former_reward_R = float(env_reward)
            current_board_state_S_prime = self.get_board_state_from_env_obs(env_observation)
            current_pie_state_P_prime = self.get_pie_rule_state_from_env_obs(env_observation)
            Q_S_A = self.read_Q_value_table(self._former_board_state_S, self._former_pie_rule_state_P, self._former_action_A)
            max_a = self.max_Q_action(current_board_state_S_prime, current_pie_state_P_prime)
            max_a_Q_S_prime_a = self.read_Q_value_table(self._former_board_state_S, self._former_pie_rule_state_P, max_a)
            new_Q_S_A = Q_S_A + self._alpha * (self._former_reward_R + self._gamma * max_a_Q_S_prime_a - Q_S_A)
            self.update_Q_value_table(self._former_board_state_S, self._former_pie_rule_state_P, self._former_action_A, new_Q_S_A)
            self._former_board_state_S = current_board_state_S_prime
            self._former_pie_rule_state_P = current_pie_state_P_prime

        # Call epislon-greedy method to get the next action given the current Q_values
        self._former_action_A = self.get_epsilon_greedy_action(self._former_board_state_S, self._former_pie_rule_state_P)
        self._actions_taken += 1
        return(self._former_action_A)

    # Function to load from file Q_values from past game plays and store it in Q_values dictionary
    def load_past_experience(self) -> bool:
        # choose the appropriate file of Q_values based on whether environment is dense or sparse with rewards
        if self._env_sparse_rewards == False:
            path = './basic_agent_experience_01.json'
        else:
            path = './basic_agent_experience_sparse_01.json'

        path_obj = Path(path)
        ret_val = False
        if path_obj.exists():
            experience_file_json = open(path, "r")
            tmp_Q_value_table = json.load(experience_file_json)
            for state_keys in tmp_Q_value_table.keys():
                for act_str_keys in tmp_Q_value_table[state_keys].keys():
                    self._Q_value_table[state_keys] = dict()
                    self._Q_value_table[state_keys][int(act_str_keys)] = tmp_Q_value_table[state_keys][act_str_keys]
            experience_file_json.close()
            ret_val = True
        else:
            ret_val = False
        return ret_val

    # Function to store the current Q_values to a file to use for future gaames as JSON files
    def store_past_experience(self):
        json_object = json.dumps(self._Q_value_table, indent = 4)

        # Save to appropriate file based on sparse or dense environment
        if self._env_sparse_rewards == False:
            path = './basic_agent_experience_01.json'
        else:
            path = './basic_agent_experience_sparse_01.json'

        with open(path, "w") as outfile:
            outfile.write(json_object)
        pass