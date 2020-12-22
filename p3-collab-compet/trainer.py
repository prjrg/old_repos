import numpy as np
from collections import deque
import torch


class Trainer:
    def __init__(self, env, maddpg_agents, brain_name, n_agents, n_episodes=5000, max_t=2000):
        """
        Params:
            env: environment to solve
            maddpg_agents: agents to train
            brain_name: the brain considered for training
            n_agents: the number of agents trained simultaneously
            n_episodes: the number of episodes we will train
            max_t: maximum number of steps considered in a continuous episode
        """
        # Initialize class variables neccessary for the training method
        self.max_t = max_t
        self.env = env
        self.maddpg_agents = maddpg_agents
        self.episodes = n_episodes
        self.brain_name = brain_name
        self.n_agents = n_agents
        
    def train(self, print_every):
        # Set local variables from class variables
        max_t = self.max_t
        env = self.env
        maddpg_agents = self.maddpg_agents
        episodes = self.episodes
        brain_name = self.brain_name
        n_agents = self.n_agents
        
        # Deques for the scores values, moving average and mean over 100 episodes
        scores_deque = deque(maxlen=100)
        scores_moving_avg = []
        scores_max = []
        
        for episode in range(episodes):
            # Reset the environment
            maddpg_agents.reset()
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations
            scores = np.zeros((n_agents,))
            
            # Loop over an episode
            for _ in range(max_t):
                # Predict the best actions for the current states
                actions = maddpg_agents.act(states)
                
                env_info = env.step(actions)[brain_name]
                
                # Predict next states, rewards, dones and compute current scores
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done
                
                # Update the memory replay of the agent with every result
                maddpg_agents.step(states, actions, rewards, next_states, dones)
                
                # Add the rewards to the scores
                scores += np.array(np.asarray(rewards))
                
                states = next_states
               
                if np.any(dones):
                    break
            
            score = np.max(scores)
            scores_max.append(score)
            scores_deque.append(score)
            scores_moving_avg.append(np.mean(scores_deque))
            
            # Save networks to path if the environment has been solved
            if scores_moving_avg[-1] >= 0.9:
                print('Environment solved in {} episodes'.format(episode-99))
                self.__save_net__(maddpg_agents)
                return scores_max, scores_moving_avg
                
            # Print every print_every the value of the scores
            if episode % print_every == 0:
                print('\rEpisode [{}/{}]\tScores: {:,.2f}\tMoving Average: {:,.2f}'.format(episode + 1, episodes, scores_max[-1], scores_moving_avg[-1])) 
                
        return scores_max, scores_moving_avg
                

    def __save_net__(self, maddpg_agents):
        torch.save(maddpg_agents.get(0).actor_local.state_dict(), 'agent0_actor.pth')
        torch.save(maddpg_agents.get(0).critic_local.state_dict(), 'agent0_critic.pth')
        torch.save(maddpg_agents.get(1).actor_local.state_dict(), 'agent1_actor.pth')
        torch.save(maddpg_agents.get(1).critic_local.state_dict(), 'agent1_critic.pth')
        
