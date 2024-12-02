import random
import logging
from ourhexenv import OurHexGame
from dqnAgent import load_dqn_agent
from ourhexgame.agent_group4.g04agent import load_mcts_agent

# Setup logging
logging.basicConfig(filename="hex_game_results.log", level=logging.INFO, format="%(message)s")
logging.info("Starting Hex Game Simulations")

def run_hex_games(num_games=50):
    # Initialize environment
    board_size = 11  # Set the board size
    env = OurHexGame(board_size=board_size, sparse_flag=False)
    
    # Load agents
    dqnAgent = load_dqn_agent(env, "dqn_agent.pt")  # Ensure correct agent loading
    mctsAgent = load_mcts_agent(env, "mcts_agent.pt")  # Ensure correct agent loading
    
    results = {"DQNAgent": 0, "MCTSAgent": 0, "Draws": 0}
    
    for game_num in range(1, num_games + 1):
        env.reset()
        done = False
        winner = None
        
        while not done:
            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()
                
                if termination or truncation:
                    if termination:
                        winner = agent  # Assign the winning agent
                    done = True
                    break
                
                # Select action based on the current agent
                if agent == "player_2":
                    action = mctsAgent.select_action(observation, agent, reward, termination, truncation)
                else:
                    action = dqnAgent.select_action(observation, agent)
                
                # Step the environment
                env.step(action)
                
                # DQNAgent training step
                if agent == "player_1":
                    # Preprocess the current and next observations
                    state_tensor = dqnAgent.preprocess_observation(observation, agent)
                    next_observation, next_reward, next_termination, next_truncation, next_info = env.last()
                    next_state_tensor = dqnAgent.preprocess_observation(next_observation, agent)
                    dqnAgent.store_experience(state_tensor, action, next_reward, next_state_tensor, next_termination or next_truncation)
                    dqnAgent.train_step()
        
        # Log the result of the game
        if winner == "player_1":
            results["DQNAgent"] += 1
            logging.info(f"Game {game_num}: Winner - DQNAgent (Red)")
        elif winner == "player_2":
            results["MCTSAgent"] += 1
            logging.info(f"Game {game_num}: Winner - MCTSAgent (Blue)")
        else:
            results["Draws"] += 1
            logging.info(f"Game {game_num}: Draw")
        
        print(f"Game {game_num} completed.")
    
    # Print final results
    print("\nSimulation Results:")
    print(f"Total games: {num_games}")
    print(f"DQNAgent wins: {results['DQNAgent']}")
    print(f"MCTSAgent wins: {results['MCTSAgent']}")
    print(f"Draws: {results['Draws']}")
    
    # Log final results
    logging.info("\nSimulation Results:")
    logging.info(f"Total games: {num_games}")
    logging.info(f"DQNAgent wins: {results['DQNAgent']}")
    logging.info(f"MCTSAgent wins: {results['MCTSAgent']}")
    logging.info(f"Draws: {results['Draws']}")

if __name__ == "__main__":
    run_hex_games(num_games=50)
