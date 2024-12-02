from ourhexenv import OurHexGame
from g10agent import G10Agent
from gYYagent import GYYAgent
import random

# Initialize the environment
env = OurHexGame(board_size=11)
env.reset()

# Initialize agents
gXXagent = G10Agent(env)  # Player 1
gYYagent = GYYAgent(env)  # Player 2

# Define the number of episodes
episodes = 10

# Loop through episodes
for episode in range(1, episodes + 1):
    env.reset()
    print(f"\nStarting Episode {episode}/{episodes}")
    done = False

    while not done:
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                if termination:
                    if reward > 0:
                        print(f"Winner: {agent} (Player {1 if agent == 'player_1' else 2})")
                    elif reward < 0:
                        print(f"Winner: {'player_2' if agent == 'player_1' else 'player_1'} (Player {2 if agent == 'player_1' else 1})")
                    else:
                        print("It's a draw!")
                done = True
                break

            # Decide action based on the agent
            if agent == "player_1":
                action = gXXagent.select_action(observation, reward, termination, truncation, info)
            else:
                action = gYYagent.select_action(observation, reward, termination, truncation, info)

            # Step in the environment
            env.step(action)

    # Replay and save the GXX agent after each episode
    gXXagent.replay()
    gXXagent.save()

print("\nTraining Completed!")
env.close()
