"""
yes
"""

import math
from copy import deepcopy
from ourhexenv import OurHexGame
# from agents.g1agent import G01Agent
# from agents.g2agent import G02Agent
# from agents.g3agent import DQNAgent
from g02agent import UCBDQNAgent


# env = OurHexGame(board_size=11, render_mode="human")
sparse_flag = False
env = OurHexGame(board_size=11, sparse_flag=sparse_flag, render_mode="human")
env.reset(seed=42)

# n_actions = env.action_space("player_1").n
# n_observations = env.board_size * env.board_size + 1  # +1 for pie rule

# Configure agents
# g1agent = G01Agent(env, "player_1")
# g1agent = DQNAgent(env, "player_1")
g1agent = UCBDQNAgent(env)

# g2agent = G02Agent(env, "player_2")
# g2agent = G01Agent(env, "player_2")
g2agent = UCBDQNAgent(env)


agents = {
    'player_1': g1agent,
    'player_2': g2agent
}

# train dqn agent
num_episodes = 10000
# g1agent.train(agents, num_episodes)

# g1agent.plot_rewards()

# evaluate dqn agent
# filename = f"./ucb_models_dense/ucb_dqn_agent_e{num_episodes}.pt"
# filename = "./ucb_models/ucb_dqn_agent_e3300.pt"
model_path = "sparse_ucb_dqn_agent.pt" if sparse_flag else "dense_ucb_dqn_agent.pt"
num_episodes = 1
rpe = g1agent.evaluate(agents, num_episodes,
                       model_path=model_path)  # rewards per episode

if sparse_flag:
    total_wins = sum(1 for r in rpe if r == 1)
    total_losses = sum(1 for r in rpe if r == -1)
else:
    min_reward = -math.ceil((env.board_size * env.board_size) / 2)
    total_wins = sum(1 for r in rpe if r > min_reward)
    total_losses = sum(1 for r in rpe if r <= min_reward)

print(f"win:lose = {total_wins}:{total_losses}")
print(f"Average win rate: {total_wins / num_episodes}")
print(f"Average reward: {sum(rpe) / num_episodes}")

env.close()
