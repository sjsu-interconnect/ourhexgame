import random
import logging
from ourhexenv import OurHexGame
from dqnAgent import DQNAgent
from g04agent import MCTSAgent

# Setup logging
logging.basicConfig(filename="hex_game_results_hyperparam.log", level=logging.INFO, format="%(message)s")

# Define hyperparameter ranges for MCTSAgent
MCTS_HYPERPARAMS = {
    "num_simulations": {"type": "int", "range": [10, 150]},
    "c_puct": {"type": "float", "range": [0.5, 2.0]},
    "dirichlet_alpha": {"type": "float", "range": [0.1, 0.3]},
    "epsilon": {"type": "float", "range": [0.1, 0.4]}
}

def sample_mcts_hyperparameters(hyperparams):
    sampled = {}
    for param, details in hyperparams.items():
        if details["type"] == "int":
            sampled[param] = random.randint(details["range"][0], details["range"][1])
        elif details["type"] == "float":
            sampled[param] = random.uniform(details["range"][0], details["range"][1])
        else:
            raise ValueError(f"Unsupported type for hyperparameter {param}")
    return sampled

# Initialize DQNAgent once, as its configuration remains constant
board_size = 11  # Set the board size
env = OurHexGame(board_size=board_size, sparse_flag=False)
dqnAgent = DQNAgent(env)

def run_hex_games(dqn_agent, mcts_agent, num_games=20, trial_num=1):
    """
    Runs a series of Hex games between the DQNAgent and MCTSAgent.

    Args:
        dqn_agent (DQNAgent): The DQN agent.
        mcts_agent (MCTSAgent): The MCTS agent with specific hyperparameters.
        num_games (int): Number of games to simulate.
        trial_num (int): Current trial number for logging purposes.
    """
    # Log the start of a new trial with its hyperparameters
    logging.info(f"\n--- Trial {trial_num} ---")
    logging.info(f"MCTS Hyperparameters: num_simulations={mcts_agent.num_simulations}, "
                 f"c_puct={mcts_agent.c_puct}, dirichlet_alpha={mcts_agent.dirichlet_alpha}, "
                 f"epsilon={mcts_agent.epsilon}")
    print(f"\nStarting Trial {trial_num} with MCTS Hyperparameters: {MCTS_HYPERPARAMS}")

    # Initialize environment for the trial
    results = {"DQNAgent": 0, "MCTSAgent": 0, "Draws": 0}

    for game_num in range(1, num_games + 1):
        env.reset()
        done = False
        winner = None
        last_agent = None  # Track the last agent who made a move

        while not done:
            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()

                if termination or truncation:
                    if termination:
                        winner = last_agent  # Assign winner based on last_agent
                    done = True
                    break

                # Select action based on the current agent
                if agent == "player_2":
                    action = mcts_agent.select_action(observation, reward, termination, truncation, info)
                else:
                    action = dqn_agent.select_action(observation, agent)

                # Step the environment
                env.step(action)

                # Update last_agent after a successful move
                last_agent = agent

                # DQNAgent training step
                if agent == "player_1":
                    # Preprocess the current and next observations
                    state_tensor = dqn_agent.preprocess_observation(observation, agent)
                    next_observation, next_reward, next_termination, next_truncation, next_info = env.last()
                    next_state_tensor = dqn_agent.preprocess_observation(next_observation, agent)
                    dqn_agent.store_experience(state_tensor, action, next_reward, next_state_tensor, next_termination or next_truncation)
                    dqn_agent.train_step()

        # Log the result of the game
        if winner == "player_1":
            results["DQNAgent"] += 1
            logging.info(f"Game {game_num}: Winner - DQNAgent (RED)")
        elif winner == "player_2":
            results["MCTSAgent"] += 1
            logging.info(f"Game {game_num}: Winner - MCTSAgent (BLUE)")
        else:
            results["Draws"] += 1
            logging.info(f"Game {game_num}: Draw")

        print(f"Game {game_num} completed.")

    # Print and log final results for the trial
    print("\nTrial Results:")
    print(f"Total games: {num_games}")
    print(f"DQNAgent wins: {results['DQNAgent']}")
    print(f"MCTSAgent wins: {results['MCTSAgent']}")
    print(f"Draws: {results['Draws']}")

    logging.info("\nSimulation Results:")
    logging.info(f"Total games: {num_games}")
    logging.info(f"DQNAgent wins: {results['DQNAgent']}")
    logging.info(f"MCTSAgent wins: {results['MCTSAgent']}")
    logging.info(f"Draws: {results['Draws']}")


def main():
    num_iterations = 20  # Number of random hyperparameter trials

    for trial in range(1, num_iterations + 1):
        print(f"\n=== Starting Trial {trial} ===")
        
        # Sample hyperparameters for MCTSAgent
        sampled_hyperparams = sample_mcts_hyperparameters(MCTS_HYPERPARAMS)
        
        # Initialize a new MCTSAgent with sampled hyperparameters
        mctsAgent = MCTSAgent(
            env=env,
            num_simulations=sampled_hyperparams["num_simulations"],
            c_puct=sampled_hyperparams["c_puct"],
            dirichlet_alpha=sampled_hyperparams["dirichlet_alpha"],
            epsilon=sampled_hyperparams["epsilon"]
        )
        
        # Run Hex games for the current trial
        run_hex_games(dqn_agent=dqnAgent, mcts_agent=mctsAgent, num_games=20, trial_num=trial)

if __name__ == "__main__":
    main()
