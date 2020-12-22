from collections import deque
import random
import numpy as np

class ExperienceReplayBuffer:
    def __init__(self, batch_size=8, episode_seq=8, buffer_size=1000, min_episodes=3):
        self.buffer = deque([[]], maxlen=1000)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.episode_seq = episode_seq
        self.min_episodes = 3

    def add(self, state, action, reward, next_state, done):
        if done == 1:
            new_episode = [(state, action, reward, next_state, done), ]
            self.buffer.append(new_episode)
        else:
            self.buffer[-1].append((state, action, reward, next_state, done))

    def sample(self):
        sampled_episodes = random.sample(list(self.buffer), self.batch_size)
        samples_state = []
        samples_action = []
        samples_reward = []
        samples_next_state = []
        samples_done = []

        for episode in sampled_episodes:
            if len(episode) + 1 - self.episode_seq >= self.episode_seq:
                first_step = np.random.randint(0, len(episode) + 1 - self.episode_seq)
                sample = episode[first_step:first_step + self.episode_seq]
                samples_state.append([s[0] for s in sample])
                samples_action.append([s[1] for s in sample])
                samples_reward.append([s[2] for s in sample])
                samples_next_state.append([s[3] for s in sample])
                samples_done.append([s[4] for s in sample])

        return samples_state, samples_action, samples_reward, samples_next_state, samples_done

    def size(self):
        return len(self.buffer)

    def is_available(self):
        return self.size() >= self.min_episodes


