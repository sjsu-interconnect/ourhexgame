from ourhexenv import OurHexGame
from g12agent import G12Agent, save_model, load_model
from dumbagent import DumbAgent

env = OurHexGame(board_size=11, render_mode="print", sparse_flag=False)
env.reset()

# Player 1
ppo_agent = G12Agent(env)

# Player 2
dumb_agent = DumbAgent(env)

# Training on dumb agent
for episode in range(2):
    env.reset()
    done = False

    while not done:
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                done = True
                break

            if agent == "player_1":
                action = dumb_agent.select_action(observation, reward, termination, truncation, info)
            else:
                action = ppo_agent.select_action(observation, reward, termination, truncation, info)
                ppo_agent.store_transition(observation["observation"].flatten(), action, reward)
                
            env.step(action)
            env.render()

    # Update PPO agent after each episode
    ppo_agent.update()

print("Training completed!")
save_path = "g12agent.pth"
save_model(ppo_agent, save_path)

env.reset()
ppo_agent_2 = G12Agent(env)
# Training on itself
for episode in range(100000):
    if episode % 100 == 0:
        print(f"Episode {episode}")
    env.reset()
    done = False

    while not done:
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                done = True
                break

            if agent == "player_1":
                action = ppo_agent_2.select_action(observation, reward, termination, truncation, info)
                ppo_agent_2.store_transition(observation["observation"].flatten(), action, reward)
            else:
                action = ppo_agent.select_action(observation, reward, termination, truncation, info)
                ppo_agent.store_transition(observation["observation"].flatten(), action, reward)
                
            env.step(action)
            # env.render()

    # Update PPO agent after each episode
    ppo_agent.update()
    ppo_agent_2.update()
    
    if episode % 1000 == 0:
        save_path = f"g12agent_{episode}.pth"
        save_model(ppo_agent, save_path)
        save_model(ppo_agent_2, save_path)

print("Training completed!")
save_path = "g12agent.pth"
save_model(ppo_agent, save_path)

# Example loading agent
ppo_agent_loaded = G12Agent(env)
load_model(ppo_agent_loaded, save_path)

done = False
while not done:
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            done = True
            break

        if agent == 'player_1':
            action = dumb_agent.select_action(observation, reward, termination, truncation, info)
        else:
            action = ppo_agent_loaded.select_action(observation, reward, termination, truncation, info)

        env.step(action)
        env.render()