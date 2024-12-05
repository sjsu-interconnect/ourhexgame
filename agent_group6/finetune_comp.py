import torch
from ourhexenv import OurHexGame
from agents.ppo_agent import Agent
from tqdm import tqdm
from agents.random_agent import RandomAgent
from agents.bit_smarter_agent import BitSmartAgent
from agent_group3.g03agent import G03Agent  # Import the G03Agent
from agent_group5.g05agent import G05Agent

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
    Train a PPO agent (Agent1) against a specified opponent (Agent2).

    Args:
        env: The environment for training.
        agent1: The PPO agent being trained.
        agent2: The opponent agent.
        episodes: Total number of training episodes.

    Returns:
        None
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
            observation, reward, terminated, truncated, info = env.last()
            done = terminated or truncated

            if not done:
                obs_flat = observation["observation"].flatten()

                # Determine current agent
                current_agent = p1_agent if agent_id == "player_1" else p2_agent

                if isinstance(current_agent, Agent):  # PPO/self-play
                    try:
                        action, probs, value = current_agent.choose_action(obs_flat, info)
                    except:
                        # import ipdb; ipdb.set_trace()
                        env.step(None)
                    env.step(action.item())
                    updated_reward = env.rewards[agent_id]
                    if current_agent == agent1:
                        agent1.remember(obs_flat, action, probs, value, updated_reward, done)
                    scores[agent_id] += updated_reward
                else:
                    action = current_agent.select_action(observation, reward, terminated, truncated, info)
                    env.step(action)
                    updated_reward = env.rewards[agent_id]
                    scores[agent_id] += updated_reward
            else:
                env.step(None)
            terminations = env.terminations

        # Train PPO Agent after each episode
        agent1.learn()


def main():
    # Initialize the environment
    env = OurHexGame(board_size=11, sparse_flag=False, render_mode=None)

    # Initialize PPO Agent with fixed parameters
    ppo_agent = Agent(
        n_actions=env.action_spaces[env.possible_agents[0]].n,
        input_dims=[env.board_size * env.board_size],
        gamma=0.99,
        actor_lr=0.0005,
        critic_lr=0.0005,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10
    )

    # Load pre-trained model for self-play
    print("\nLoading self-play checkpoint...")
    load_ppo_checkpoint(ppo_agent, filename='ppo_checkpoint_after_selfplay.pth')

    # Load G03Agent and its model
    print("\nLoading G03Agent...")
    ppo_agent_g03 = G05Agent(env)
    # MODEL_PATH_NEW = "agent_group3/trained_dense_agent.pth"
    # ppo_agent_g03.load_model(MODEL_PATH_NEW)

    # Train PPO agent against competing agent
    print("\nTraining against competing Agent...")
    train_against_agent(env, ppo_agent, ppo_agent_g03, episodes=2500)

    # Save the final model
    save_ppo_checkpoint(ppo_agent, filename='ppo_checkpoint_final.pth', iteration=2500)

    print("Training completed.")


if __name__ == "__main__":
    main()