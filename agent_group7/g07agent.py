from typing import Any
import os
from ourhexenv import OurHexGame

from agent_group7.protocols import Agent

from agent_group7.models import PPOAgent



class G07Agent(Agent):
    """Wrapper class for the two agents.
    """
    def __init__(self, env: OurHexGame) -> "G07Agent":
        # Figure out the path of this file
        self.directory = os.path.dirname(os.path.abspath(__file__))
        env_name_expected_file = [
            ("G07AGENT_SPARSE", "sparse_agent.pth"),
            ("G07AGENT_DENSE", "dense_agent.pth"),
        ]
        for env_name, expected_file in env_name_expected_file:
            if os.environ.get(env_name, None) is None:
                file_path = os.path.join(self.directory, expected_file)
                if os.path.exists(file_path):
                    os.environ[env_name] = file_path
        if env.sparse_flag:
            agent_file = os.environ.get("G07AGENT_SPARSE", None)
            if not agent_file:
                raise ValueError("G07AGENT_SPARSE environment variable not set")
            self.agent = PPOAgent.from_file(agent_file, env=env)
        else:
            agent_file = os.environ.get("G07AGENT_DENSE", None)
            if not agent_file:
                raise ValueError("G07AGENT_DENSE environment variable not set")
            self.agent = PPOAgent.from_file(agent_file, env=env)
            self.env = env

    def select_action(self, observation, reward, termination, truncation, info):
        return self.agent.select_action(observation, reward, termination, truncation, info)
