import math
import random

import gym
import torch
import torch.nn.functional as F
from torch import nn

from Model import DeepRecurrentQN
from ReplayBuffer import ExperienceReplayBuffer
from TwoModelSingleGPU import TwoHeadNet
from optimizer_ranger_with_gc import Ranger
import numpy as np

GAMMA = 0.9
TAU = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.state_size = state_size

        self.main_network = DeepRecurrentQN(action_size)
        self.target_network = DeepRecurrentQN(action_size)

        self.twohead = TwoHeadNet(self.main_network, self.target_network, device).to(device)

        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()

        self.optimizer = Ranger(self.main_network.parameters(), lr=1e-4)

        self.episode_steps = 5
        self.batch_size = 1
        self.num_episodes = 1000
        self.min_episodes = 3
        self.replay = ExperienceReplayBuffer(self.batch_size, self.episode_steps, self.num_episodes, self.min_episodes)

    def img_to_tensor(self, img):
        img_tensor = torch.FloatTensor(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor

    def img_list_to_batch(self, x):
        temp_batch = self.img_to_tensor(x[0])
        temp_batch = temp_batch.unsqueeze(0)
        for i in range(1, len(x)):
            img = self.img_to_tensor(x[i])
            img = img.unsqueeze(0)
            temp_batch = torch.cat([temp_batch, img], dim=0)
        return temp_batch

    def train(self):
        if self.replay.is_available():
            sample = self.replay.sample()
            states, actions, rewards, next_states, dones = sample

            states = self.img_list_to_batch(states).to(device)
            next_states = self.img_list_to_batch(next_states).to(device)
            actions = torch.from_numpy(np.vstack(actions)).long().to(device)
            rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)

            expected_actions, _ = self.twohead(next_states).detach().max(1)[1].unsqueeze(1)
            Q_next, _ = self.twohead(next_states, b=False).gather(1, expected_actions)

            Q_est = Q_next.detach().clone()
            Q_est[:, 0, :] = rewards + (GAMMA * Q_next[:, 0, :])

            Q_expected = self.twohead(states)[0].gather(1, actions)

            self.optimizer.zero_grad()
            loss = F.smooth_l1_loss(Q_est, Q_expected)
            loss.backward()
            for param in self.main_network.parameters():
                param.grad.data.clamp(-1, 1)
            self.optimizer.step()

            self.soft_update(self.main_network, self.target_network, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def act(self, obs, hidden, epsilon):
        state = self.img_to_tensor(obs).unsqueeze(0).to(device)
        q, new_hidden = self.twohead(state, hidden)
        if random.random() > epsilon:
            action = q[0].max(1)[1].data[0].item()
        else:
            action = random.randint(0, self.action_size - 1)
        return action, new_hidden

    def get_decay(self, epi_iter):
        decay = math.pow(0.99, epi_iter)
        if decay < 0.05:
            decay = 0.05
        return decay


if __name__ == '__main__':
    random.seed()
    env = gym.make('MsPacman-v0')
    env.reset()
    agent = Agent(env.observation_space.shape, env.action_space.n)
    i = 0
    for episode in range(3000):
        state = env.reset()
        hidden = DeepRecurrentQN.hidden_initialization(device)
        while True:
            env.render()
            action, hidden = agent.act(state, hidden, agent.get_decay(episode))
            next_state, reward, done, info = env.step(action)
            agent.replay.add(state, action, reward, next_state, done)

            if done:
                break
            if i % 4 == 0:
                agent.train()
            i += 1