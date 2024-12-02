import torch

from agent_group11.dqn_noisy import DQNAgent


def load_model(agent, filename="dqn_model.pth"):
    agent.policy_net.load_state_dict(torch.load(filename, weights_only=True))
    # Set the model to evaluation mode
    agent.policy_net.eval()
    print(f"Model loaded from {filename}")


class G11Agent:
    def __init__(self, env):
        self.env = env
        self.agent_name = 'player_1'
        self.dqn_agent = DQNAgent(env, buffer_size=10000, batch_size=128, name=self.agent_name)
        load_file = "agent_group11/saved_models/dqn_noisy_against_smart_" + self.agent_name
        load_model(self.dqn_agent, load_file)

    def select_action(self, observation, reward, termination, truncation, info):
        return self.dqn_agent.select_action(observation, reward, termination, truncation, info)