import click
from agent_group7.models import TrainablePPOAgent, ActorCriticNN, PPOAgent
from agent_group7.protocols import Agent

from ourhexenv import OurHexGame
import torch

import random
from copy import deepcopy

import time
import ray
from ray import tune
from ray import train
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from pathlib import Path
import numpy as np

from typing import Tuple


class RandomAgent(Agent):
    def select_action(self, observation, reward, termination, truncation, info):
        actions = np.arange(0, 122)
        mask = info["action_mask"]
        valid_actions = [a for a in actions if mask[a]]
        return np.random.choice(valid_actions)


@click.group()
def cli():
    pass


class SelfPlayTrainable(ray.tune.Trainable):

    def setup(self, config: dict):
        self.lr_actor = config["lr_actor"]
        self.lr_critic = config["lr_critic"]
        self.swap_rate = config["swap_rate"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = OurHexGame(sparse_flag=self.config["sparse_flag"], render_mode=None)
        self.optimize_policy_epochs = config["optimize_policy_epochs"]
        self.target_agent = TrainablePPOAgent(
            self.env,
            ActorCriticNN(),
            device=self.device,
            lr_actor=self.lr_actor,
            lr_critic=self.lr_critic
        )
        self.oponent_agent = RandomAgent(self.env)
        self.target_agent_score = 1400
        self.oponent_agent_score = 1400
        self.batch_size = config["batch_size"]
        self.play_style = "random"
        self.target_agent_wins = 0
        self.games_played = 0

    def step(self):
        running_reward = 0
        target_agent_wins = 0
        for _ in range(self.config["games_per_step"]):
            self.games_played += 1
            # The swap rate is the probability of swapping the agents
            if random.random() < self.swap_rate:
                player_1_agent = self.target_agent
                player_2_agent = self.oponent_agent
                player_name_to_rating = {
                    "player_1": self.target_agent_score,
                    "player_2": self.oponent_agent_score
                }
                player_name_to_agent_name = {
                    "player_1": "target",
                    "player_2": "oponent"
                }
            else:
                player_1_agent = self.oponent_agent
                player_2_agent = self.target_agent
                player_name_to_rating = {
                    "player_1": self.oponent_agent_score,
                    "player_2": self.target_agent_score
                }
                player_name_to_agent_name = {
                    "player_1": "oponent",
                    "player_2": "target"
                }
            
            player_1_cumulative_reward, player_2_cumulative_reward, winner = self.play_episode(
                player_1_agent, player_2_agent)
            # get only the reward from the perspective of the target agent
            if player_name_to_agent_name["player_1"] == "target":
                running_reward += player_1_cumulative_reward
            else:
                running_reward += player_2_cumulative_reward
            # calculate win_rate
            if player_name_to_agent_name[winner] == "target":
                target_agent_wins += 1
                self.target_agent_wins += 1
            # Calculate the new ELO rating for the agents
            if winner == "player_1":
                self.target_agent_score = self.calculate_elo_rating(
                    self.target_agent_score,
                    player_name_to_rating["player_2"],
                    1
                )
                self.oponent_agent_score = self.calculate_elo_rating(
                    self.oponent_agent_score,
                    player_name_to_rating["player_1"],
                    0
                )
            elif winner == "player_2":
                self.target_agent_score = self.calculate_elo_rating(
                    self.target_agent_score,
                    player_name_to_rating["player_2"],
                    0
                )
                self.oponent_agent_score = self.calculate_elo_rating(
                    self.oponent_agent_score,
                    player_name_to_rating["player_1"],
                    1
                )

        target_loss = self.target_agent.optimize_policy(self.optimize_policy_epochs, self.batch_size)
        if self.play_style == "self-play":
            self.target_agent.buffer.clear()
        if self.training_iteration % 100 == 0:
            overall_win_rate = self.target_agent_wins/self.games_played
            if overall_win_rate > 0.9:
                self.play_style = "self-play"
                self.oponent_agent = TrainablePPOAgent(
                    self.env,
                    ActorCriticNN(),
                    device=self.device
                )
                self.oponent_agent.nn.load_state_dict(deepcopy(self.target_agent.nn.state_dict()))
                self.oponent_agent.old_nn.load_state_dict(deepcopy(self.target_agent.old_nn.state_dict()))
                self.oponent_agent.optimizer.load_state_dict(deepcopy(self.target_agent.optimizer.state_dict()))
        return {"average_reward": running_reward/self.config["games_per_step"], 
                "target_agent_score": self.target_agent_score,
                "target_agent_win_rate": target_agent_wins/self.config["games_per_step"],
                "target_loss": target_loss,
                }
    
    def calculate_elo_rating(self, current_rating: float, opponent_rating: float, score: float, k: int = 32) -> float:
        expected_score = 1 / (1 + 10 ** ((opponent_rating - current_rating) / 400))
        return current_rating + k * (score - expected_score)
            

    def play_episode(
            self,
            player_1_agent: TrainablePPOAgent,
            player_2_agent: TrainablePPOAgent
        ) -> Tuple[int, int, str]:
        """Play an episode of the game.

        Args:
            player_1_agent (Agent): player 1
            player_2_agent (Agent): player 2

        Returns:
            Tuple[int, int, str]: (player1_cumulative_reward, player2_cumulative_reward, winner)
        """
        env = self.target_agent.env
        env.reset()
        episode_cumulative_reward_per_agent = {
            "player_1": 0,
            "player_2": 0
        }
        agent_names_to_agents = {
            "player_1": player_1_agent,
            "player_2": player_2_agent
        }
        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            episode_cumulative_reward_per_agent[agent_name] += reward
            agent = agent_names_to_agents[agent_name]
            if termination or truncation:
                action = None
            else:
                action = agent.select_action(
                    observation,
                    reward,
                    termination,
                    truncation,
                    info
                )
            env.step(action)

        if episode_cumulative_reward_per_agent["player_1"] > episode_cumulative_reward_per_agent["player_2"]:
            winner = "player_1"
        else:
            winner = "player_2"
        return (episode_cumulative_reward_per_agent["player_1"],
                episode_cumulative_reward_per_agent["player_2"],
                winner)
    
    def save_checkpoint(self, checkpoint_dir: str):
        checkpoint_data = {
            "nn_state_dict": self.target_agent.nn.state_dict(),
            "old_nn_state_dict": self.target_agent.old_nn.state_dict(),
            "optimizer_state_dict": self.target_agent.optimizer.state_dict(),
            "target_agent_score": self.target_agent_score,
            "oponent_agent_score": self.oponent_agent_score,
            "play_style": self.play_style,
            "target_agent_wins": self.target_agent_wins,
            "games_played": self.games_played
        }
        if self.play_style == "random":
            opponnent_nn_data = {}
        else:
            opponnent_nn_data = {
                "oponent_nn_state_dict": self.oponent_agent.nn.state_dict(),
                "oponent_old_nn_state_dict": self.oponent_agent.old_nn.state_dict(),
                "oponent_optimizer_state_dict": self.oponent_agent.optimizer.state_dict()
            }
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "wb") as fp:
            all_data = {**checkpoint_data, **opponnent_nn_data}
            pickle.dump(all_data, fp)

        return Checkpoint.from_directory(checkpoint_dir)

    def load_checkpoint(self, checkpoint_dir: str):
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            checkpoint_state = pickle.load(fp)
        self.target_agent.nn.load_state_dict(checkpoint_state["nn_state_dict"])
        self.target_agent.old_nn.load_state_dict(
            checkpoint_state["old_nn_state_dict"])
        self.target_agent.optimizer.load_state_dict(
            checkpoint_state["optimizer_state_dict"])
        self.target_agent_score = checkpoint_state["target_agent_score"]
        self.oponent_agent_score = checkpoint_state["oponent_agent_score"]
        self.play_style = checkpoint_state["play_style"]
        self.target_agent_wins = checkpoint_state["target_agent_wins"]
        self.games_played = checkpoint_state["games_played"]
        if self.play_style != "random":
            self.oponent_agent.nn.load_state_dict(
                checkpoint_state["oponent_nn_state_dict"])
            self.oponent_agent.old_nn.load_state_dict(
                checkpoint_state["oponent_old_nn_state_dict"])
            self.oponent_agent.optimizer.load_state_dict(
                checkpoint_state["oponent_optimizer_state_dict"])


@cli.command()
@click.option("--sparse/--no-sparse", is_flag=True, default=False)
@click.option("--gpu/--no-gpu", is_flag=True, default=False)
@click.option(
    "--lr-actor",
    type=float,
    default=None
)
@click.option(
    "--lr-critic",
    type=float,
    default=None
)
@click.option(
    "--swap-rate",
    type=float,
    default=None
)
@click.option(
    "--optimize-policy-epochs",
    type=int,
    default=None
)
@click.option(
    "--num-samples",
    type=int,
    default=10
)
@click.option(
    "--games-per-step",
    type=int,
    default=None
)
@click.option(
    "--batch-size",
    type=int,
    default=None
)
@click.option(
    "--num-cpus",
    type=int,
    help="Number of CPUs to use for training. Do not set for automatic detection.",
)
@click.option(
    "--log-dir",
    envvar="TUNE_LOG_DIR",
)
@click.option(
    "--max-t",
    type=int,
    default=10000
)
def train_agent(
    sparse: bool,
    gpu: bool,
    lr_actor: float,
    lr_critic: float,
    swap_rate: float,
    optimize_policy_epochs: int,
    num_samples: int,
    games_per_step: int,
    batch_size: int,
    num_cpus: int,
    log_dir: str,
    max_t: int
    ):
    if not gpu:
        # Currently there is a bug in WSL2 that prevents Ray tune from auto-detecting
        # whether the current device is a GPU or not.
        # A quick and dirty fix is to directly inform RAY that the current device is a CPU.
        ray.init(num_gpus=0)
        print("--no-gpu flag detected. Forcing Ray to run on CPU.")
    config = {
        "lr_critic": lr_critic or tune.loguniform(1e-5, 1e-2),
        "lr_actor": lr_actor or tune.loguniform(1e-5, 1e-2),
        "swap_rate": swap_rate or tune.uniform(0.1, 0.5),
        "games_per_step": games_per_step or tune.choice([10, 20, 30, 40]),
        "optimize_policy_epochs": optimize_policy_epochs or tune.choice([1, 3, 5, 10]),
        "batch_size": batch_size or tune.choice([64, 128, 256, 512]),
        "sparse_flag": sparse,
    }
    scheduler = ASHAScheduler(
        max_t=max_t,
        metric="average_reward",
        mode="max",
        grace_period=10,
        reduction_factor=2
    )

    # detect gpus
    if gpu and torch.cuda.is_available():
        trainable = tune.with_resources(
            SelfPlayTrainable, resources={"gpu": 1}
            )
    elif num_cpus:
        trainable = tune.with_resources(
            SelfPlayTrainable, resources={"cpu": num_cpus}
            )
    else:
        trainable = SelfPlayTrainable
    # TODO: Figure out a way to better schedule the cpus
    # to avoid running out of memory.
    result = tune.run(
        trainable,
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        checkpoint_config=train.CheckpointConfig(num_to_keep=10, checkpoint_frequency=1),
        storage_path=log_dir or None
    )

    best_trial = result.get_best_trial("average_reward", "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(
        f"Best trial average reward: {best_trial.last_result['average_reward']}")

@cli.command()
def re_train():
    pass


@cli.command()
@click.option(
    "--checkpoint-file",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    required=True
)
@click.option("--sparse/--no-sparse", is_flag=True, default=False)
def test_agent(
    checkpoint_file: str,
    sparse: bool,
):
    env = OurHexGame(sparse_flag=sparse, render_mode=None)
    smart_agent: PPOAgent = PPOAgent.from_file(checkpoint_file, env=env)
    dumb_agent = RandomAgent(env)
    def play_episode(env: OurHexGame, player_1: Agent, player_2: Agent):
        env.reset()
        episode_cumulative_reward_per_agent = {
            "player_1": 0,
            "player_2": 0
        }
        agent_names_to_agents = {
            "player_1": player_1,
            "player_2": player_2
        }
        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            episode_cumulative_reward_per_agent[agent_name] += reward
            agent = agent_names_to_agents[agent_name]
            if termination or truncation:
                action = None
            else:
                action = agent.select_action(
                    observation,
                    reward,
                    termination,
                    truncation,
                    info
                )
            env.step(action)
        player_1_reward = episode_cumulative_reward_per_agent["player_1"]
        player_2_reward = episode_cumulative_reward_per_agent["player_2"]
        winner = "player_1" if player_1_reward > player_2_reward else "player_2"
        return episode_cumulative_reward_per_agent, winner
    total_wins = 0
    for g in range(100):
        print(f"Playing game: {g}")
        _, winner = play_episode(env, dumb_agent, smart_agent)
        if winner == "player_2":
            total_wins += 1
    print(f"Win rate: {total_wins/100}")



def main():
    cli()

if __name__ == "__main__":
    main()