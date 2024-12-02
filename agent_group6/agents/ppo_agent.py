import numpy as np
import torch as T
from agents.ppo_memory import PPOBufferMemory
from agents.acn import ActorCriticNetwork

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, actor_lr=0.0003, critic_lr=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=2, n_epochs=10, entropy_coef=0.01):
        
        # Initialize hyperparameters for PPO
        self.gamma = gamma  # Discount factor for future rewards
        self.policy_clip = policy_clip  # Clip value for PPO loss function
        self.n_epochs = n_epochs  # Number of training epochs per update
        self.gae_lambda = gae_lambda  # GAE lambda for advantage estimation
        self.entropy_coef = entropy_coef  # Coefficient for entropy bonus

        # Initialize the actor-critic network
        self.actor_critic = ActorCriticNetwork(n_actions, input_dims, actor_lr, critic_lr)
        self.actor = self.actor_critic.actor
        self.critic = self.actor_critic.critic

        # Initialize memory buffer for storing experiences
        self.memory = PPOBufferMemory(batch_size)
        
        self.device = "cuda:0" if T.cuda.is_available() else "cpu"
        self.actor_critic.to(self.device)

    def remember(self, state, action, probs, vals, reward, done):
        # Store experience in memory buffer
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation, info):
        # Convert observation to tensor and move to device
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.device)
        
        # Use the actor-critic network to choose an action based on the current state and action mask
        return self.actor_critic.act(state, info["action_mask"])
 
    def learn(self):
        for _ in range(self.n_epochs):
            # Generate batches from memory buffer
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = T.tensor(vals_arr).to(self.device)  # Convert value estimates to tensor on device
            
            advantage = np.zeros(len(reward_arr), dtype=np.float32)  # Initialize advantage array

            for t in range(len(reward_arr)):
                advantage[t] = reward_arr[t] - values[t]  # Calculate advantage

            advantage = T.tensor(advantage).to(self.device)  # Convert advantage to tensor on device

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.device)
                actions = T.tensor(action_arr[batch], dtype=T.long).to(self.device)

                critic_value, new_probs, entropy = self.actor_critic.evaluate(states, actions)

                critic_value = T.squeeze(critic_value)  # Remove dimensions of size 1

                prob_ratio = new_probs.exp() / old_probs.exp()  # Calculate probability ratio for PPO
                
                weighted_probs = advantage[batch] * prob_ratio  # Calculate weighted probabilities
                
                weighted_clipped_probs = T.clamp(prob_ratio,
                                                 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]  # Apply clipping
                
                actor_loss = -T.min(weighted_probs,
                                    weighted_clipped_probs).mean()  # Calculate actor loss with clipping

                entropy_loss = -self.entropy_coef * entropy.mean()
                returns = advantage[batch] + values[batch]  # Calculate returns
                
                critic_loss = (returns - critic_value) ** 2  # Calculate critic loss (MSE)
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss + entropy_loss  # Total loss is a combination of actor and critic loss
                
                self.actor_critic.actor_optimizer.zero_grad()  # Zero gradients for optimizer
                self.actor_critic.critic_optimizer.zero_grad()

                total_loss.backward()  # Backpropagate total loss
                
                self.actor_critic.actor_optimizer.step()  # Update actor network weights
                self.actor_critic.critic_optimizer.step()  # Update critic network weights

        self.memory.clear_memory()  # Clear memory buffer after learning

    def print_info(self, file=None):
        if file:
            with open(file, 'a') as f:
                f.write("Agent Information:\n")
                f.write("\nActor Network Info:\n")
            self.actor.print_info(file)
            with open(file, 'a') as f:
                f.write("\nCritic Network Info:\n")
            self.critic.print_info(file)

    def select_action(self, observation, reward, termination, truncation, info):
        action, _, _ = self.choose_action(observation["observation"].flatten(), info)
        return action.item()

    @classmethod
    def from_file(cls, filename, env):
        agent = cls(
            n_actions=env.action_spaces[env.possible_agents[0]].n,
            input_dims=[env.board_size * env.board_size],
            gamma=0.99,
            actor_lr=0.0003,
            critic_lr=0.0003,
            gae_lambda=0.95,
            policy_clip=0.2,
            batch_size=64,
            n_epochs=10
        )
        checkpoint = T.load(filename, map_location=T.device('cpu'))
        agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        agent.actor_critic.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        agent.actor_critic.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        return agent