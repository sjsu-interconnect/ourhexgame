from ourhexenv import OurHexGame
from agents.ppo_agent import Agent

class G06Agent(Agent):
    """Wrapper class for our agents.
    """
    def __init__(self, env: OurHexGame) -> "G06Agent":
        if env.sparse_flag:
            agent_file = 'ppo_checkpoint_sparse.pth'
            self.agent = Agent.from_file(agent_file, env=env)
        else:
            agent_file = 'ppo_checkpoint_dense.pth'
            self.agent = Agent.from_file(agent_file, env=env)
            self.env = env

    def select_action(self, observation, reward, termination, truncation, info):
        return self.agent.select_action(observation, reward, termination, truncation, info)