from dqnAgent import DQNAgent
from g04agent import MCTSAgent
import torch

# Create an environment
from ourhexenv import OurHexGame
env = OurHexGame(board_size=11)
env.reset()

# Instantiate DQNAgent
dqn_agent = DQNAgent(env)
mcts_agent = MCTSAgent(env)

# Save the agent
def save_dqn_agent(agent, filepath):
    torch.save(agent.policy_net.state_dict(), filepath)


save_dqn_agent(dqn_agent, "dqn_agent.pt")

def save_mcts_agent(agent, filepath, current_epoch=None):
    checkpoint = {
        'model_state_dict': agent.neural_network.state_dict(),
        'mcts_params': {
            'num_simulations': agent.num_simulations,
            'c_puct': agent.c_puct,
            'dirichlet_alpha': agent.dirichlet_alpha,
            'epsilon': agent.epsilon
        }
    }
    if current_epoch is not None:
        checkpoint['current_epoch'] = current_epoch

    torch.save(checkpoint, filepath)



save_mcts_agent(mcts_agent, "mcts_agent.pt")