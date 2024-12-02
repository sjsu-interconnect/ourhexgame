import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ourhexenv import OurHexGame
from agents.ppo_agent import Agent
from agents.random_agent import RandomAgent
from agents.bit_smarter_agent import BitSmartAgent
import torch
from tqdm import tqdm

def save_ppo_checkpoint(agent, filename='ppo_checkpoint.pth', iteration=0):
    """Save the PPO agent's state in a checkpoint file."""
    checkpoint = {
        'model_state_dict': agent.actor_critic.state_dict(),
        'actor_optimizer_state_dict': agent.actor_critic.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.actor_critic.critic_optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")


def load_ppo_checkpoint(agent, filename='ppo_checkpoint.pth'):
    """Load the PPO agent's state from a checkpoint file."""
    try:
        checkpoint = torch.load(filename)
        agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from {filename}")
    except FileNotFoundError:
        print(f"Checkpoint file not found at {filename}. Proceeding without loading.")


def train_against_agent(env, agent1, agent2, episodes):
    """
    Train a PPO agent (Agent1) against a specified opponent (Agent2) with role-swapping.
    """
    for episode in tqdm(range(episodes), desc="Training Episodes"):
        env.reset()
        terminations = {agent: False for agent in env.possible_agents}
        scores = {agent: 0 for agent in env.possible_agents}

        # Determine roles based on episode number
        if episode < episodes // 2:
            p1_agent, p2_agent = agent1, agent2  # Agent1 as player_1
        else:
            p1_agent, p2_agent = agent2, agent1  # Agent1 as player_2

        while not all(terminations.values()):
            agent_id = env.agent_selection
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation

            if not done:
                obs_flat = observation["observation"].flatten()

                # Determine current agent
                current_agent = p1_agent if agent_id == "player_1" else p2_agent

                if isinstance(current_agent, Agent):  # PPO/self-play
                    action, probs, value = current_agent.choose_action(obs_flat, info)
                    env.step(action.item())
                    updated_reward = env.rewards[agent_id]
                    # Only store memory for the PPO agent being trained
                    if current_agent == agent1:
                        agent1.remember(obs_flat, action, probs, value, updated_reward, done)
                    scores[agent_id] += updated_reward
                else:  # RandomAgent or BitSmartAgent
                    action = current_agent.select_action(env, info)
                    env.step(action)
                    updated_reward = env.rewards[agent_id]
                    scores[agent_id] += updated_reward
            else:
                env.step(None)
            terminations = env.terminations

        # Train PPO Agent after each episode
        agent1.learn()


def train_with_tune(config):
    """Training with Ray Tune for hyperparameter optimization."""
    env = OurHexGame(board_size=11, sparse_flag=False, render_mode=None)
    ppo_agent = Agent(
        n_actions=env.action_spaces[env.possible_agents[0]].n,
        input_dims=[env.board_size * env.board_size],
        **config
    )

    # Training against RandomAgent and BitSmartAgent
    opponents = [
        (RandomAgent(), 'random', 1000),
        (BitSmartAgent(), 'bitsmart', 1000)
    ]

    for opponent, _, episodes in opponents:
        train_against_agent(env, ppo_agent, opponent, episodes)

    # Self-play training
    print("\nStarting self-play training...")
    train_against_agent(env, ppo_agent, ppo_agent, episodes=5000)

    # Evaluate the agent after self-play
    eval_episodes = 100
    wins = 0
    for _ in range(eval_episodes):
        env.reset()
        done = False
        scores = {agent: 0 for agent in env.possible_agents}
        while not done:
            agent_id = env.agent_selection
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation
            if not done:
                if agent_id == "player_1":
                    obs_flat = observation["observation"].flatten()
                    action, _, _ = ppo_agent.choose_action(obs_flat, info)
                    env.step(action.item())
                    updated_reward = env.rewards[agent_id]
                    scores[agent_id] += updated_reward
                else:
                    action = BitSmartAgent().select_action(env, info)
                    env.step(action)
                    updated_reward = env.rewards[agent_id]
                    scores[agent_id] += updated_reward
            else:
                env.step(None)
        if scores["player_1"] > scores["player_2"]:
            wins += 1

    win_rate = wins / eval_episodes
    ray.train.report({"win_rate":win_rate})


def main():
    ray.init(num_gpus=1)

    # Hyperparameter tuning configuration
    config = {
        "gamma": tune.uniform(0.9, 0.99),
        "actor_lr": tune.loguniform(1e-5, 1e-2),
        "critic_lr": tune.loguniform(1e-5, 1e-2),
        "gae_lambda": tune.uniform(0.9, 1.0),
        "policy_clip": tune.uniform(0.1, 0.3),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "n_epochs": tune.randint(5, 21)
    }

    scheduler = ASHAScheduler(
        max_t=1000,
        grace_period=100,
        reduction_factor=2
    )

    tuner = tune.Tuner(
        tune.with_resources(train_with_tune, resources={"gpu": 1}),
        tune_config=tune.TuneConfig(
            metric="win_rate",
            mode="max",
            scheduler=scheduler,
            num_samples=50
        ),
        param_space=config,
    )

    results = tuner.fit()

    best_result = results.get_best_result("win_rate", "max", "last")
    print("Best trial config:", best_result.config)
    print("Best trial final win rate:", best_result.metrics['win_rate'])

    # Train final model with best hyperparameters
    env = OurHexGame(board_size=11, sparse_flag=False, render_mode=None)
    final_agent = Agent(
        n_actions=env.action_spaces[env.possible_agents[0]].n,
        input_dims=[env.board_size * env.board_size],
        **best_result.config
    )

    # Training against self-play with best hyperparameters
    print("\nFinal self-play training...")
    train_against_agent(env, final_agent, final_agent, episodes=5000)

    # Save the final model
    save_ppo_checkpoint(final_agent, filename="ppo_checkpoint_final.pth")


if __name__ == "__main__":
    main()