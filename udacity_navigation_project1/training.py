import numpy as np
from collections import deque
import torch

class Trainer:
    def __init__(self, env, agent, brain_name=None, n_episodes=1800, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        """Deep Q-Learning Training Constructor.

        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.max_t = max_t
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        if brain_name is None:
            brain_name = env.brain_names[0]
        self.brain_name = brain_name

    def train(self):
        """Deep Q-Learning training method

        """

        #Initialize variables for the function from class variables
        env = self.env
        agent = self.agent
        n_episodes = self.n_episodes
        max_t = self.max_t
        eps_start = self.eps_start
        eps_end = self.eps_end
        eps_decay = self.eps_decay
        brain_name = self.brain_name

        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start                    # initialize epsilon
        for i_episode in range(1, n_episodes+1):
            
            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations[0]
            score = 0

            for t in range(max_t):
                action = agent.act(state, eps)

                env_info = env.step(action)[brain_name]
                
                # Get state S(t+1) and Reward(t+1) and check if it finished
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]

                # This method computes the loss function and the optimizer step
                agent.step(state, action, reward, next_state, done)

                state = next_state
                score += reward
                if done:
                    break 
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window)>=16.5:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
                break
        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        return scores