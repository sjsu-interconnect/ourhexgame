# Filename: myrunner-eg.py

from ourhexenv import OurHexGame
import random
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from dqnAgent import load_dqn_agent
from g04agent import load_mcts_agent

# -----------------------  ORIGINAL MY RUNNER IN WHICH TO RUN ----------------------- #
env = OurHexGame(board_size=11)
env.reset()

dqnAgent = load_dqn_agent(env, "dqn_agent.pt")
mctsAgent = load_mcts_agent(env, "g04agent.pt")

# smart_agent_player_id = random.choice(env.agents)

done = False
winner = None
last_agent = None

while not done:
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            if termination:
                winner = last_agent  # Assign the winning agent
            done = True
            break

        
        if agent == 'player_2':
            action = mctsAgent.select_action(observation, reward, termination, truncation, info) # blue
        else:
            action = dqnAgent.select_action(observation, agent) # red

        #reward, termination, truncation, info

        env.step(action)
        env.render()
        
        last_agent = agent


        # After taking the action, observe the next state
        next_observation, next_reward, next_termination, next_truncation, next_info = env.last()

        # Store the experience in smart_agent's replay buffer
        if agent == "player_1":
            # Preprocess the current and next observations
            state_tensor = dqnAgent.preprocess_observation(observation, agent)
            next_state_tensor = dqnAgent.preprocess_observation(next_observation, agent)
            dqnAgent.store_experience(state_tensor, action, next_reward, next_state_tensor, next_termination or next_truncation)

            # Perform a training step
            dqnAgent.train_step()

# -------------------------------
# Print the winner after the game ends
# -------------------------------
if winner:
    print(f"Game over. The winner is {winner}. Close the render window manually.")
else:
    print("Game over. No winner was determined.")

# -------------------------------
# Keep the render window open until manually closed
# -------------------------------
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

env.close()