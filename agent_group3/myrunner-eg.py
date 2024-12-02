from ourhexenv import OurHexGame
from g03agent import G03Agent
import random
import pygame
import numpy as np

# Sparse True - Sparse reward, Sparse False - Dense reward
sparse = False
G03Agent.train_agent(num_episodes=20, sparse_flag = sparse)

env = OurHexGame(board_size=11, sparse_flag = sparse)
env.reset()

if random.choice([True, False]):
    gXXagent = G03Agent(env, player_id="player_1")
    gYYagent = G03Agent(env, player_id="player_2")
    print("Player 1 is the trained agent.")
else:
    gXXagent = G03Agent(env, player_id="player_2")
    gYYagent = G03Agent(env, player_id="player_1")
    print("Player 2 is the trained agent.")
if sparse:
    gXXagent.load_model("trained_sparse_agent.pth")
else:
    gXXagent.load_model("trained_dense_agent.pth")


done = False
render = True 
step_count = 0

while not done:
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            done = True
            break
        action = gXXagent.select_action(observation, reward, termination, truncation, info) \
                 if agent == 'player_1' else \
                 gYYagent.select_action(observation, reward, termination, truncation, info)
        env.step(action)

        if render and step_count % 5 == 0:  
            env.render()
        
        step_count += 1

# Print Winner
if env.rewards['player_1'] > env.rewards['player_2']:
    print("Player 1 (Red) wins!")
elif env.rewards['player_1'] < env.rewards['player_2']:
    print("Player 2 (Blue) wins!")
else:
    print("It's a draw!")
# Keep showing last step
if render:
    print("Press 'X' to close the game window.")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  
                running = False
    env.close()  
