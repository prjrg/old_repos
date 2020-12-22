import gym
import numpy as np
import ptan
from typing import List, Any, Optional, Tuple
import torch.nn as nn


class ToyEnv(gym.Env):
    def __init__(self):
        super(ToyEnv, self).__init__()
        self.observation_space = gym.spaces.Discrete(n=5)
        self.action_space = gym.spaces.Discrete(n=3)
        self.step_index = 0

    def reset(self):
        self.step_index = 0
        return self.step_index

    def step(self, action):
        is_done = self.step_index == 10
        if is_done:
            return self.step_index % self.observation_space.n, 0.0, is_done, {}
        self.step_index += 1
        return self.step_index % self.observation_space.n, float(action), self.step_index == 10, {}


class DullAgent(ptan.agent.BaseAgent):
    def __init__(self, action: int):
        self.action = action

    def __call__(self, observations: List[Any], state: Optional[List] = None) -> Tuple[List[int], Optional[List]]:
        return [self.action for _ in observations], state


env = ToyEnv()
agent = DullAgent(action=1)
exp_source = ptan.experience.ExperienceSource(env=env, agent=agent, steps_count=2)

for idx, exp in enumerate(exp_source):
    if idx > 2:
        break
    print(exp)

exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=1.0, steps_count=1)
buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=100)

print(len(buffer))


class DQNNet(nn.Module):
    def __init__(self):
        super(DQNNet, self).__init__()
        self.ff = nn.Linear(5,3)

        def forward(self, x):
            return self.ff(x)


net = DQNNet()
tgt_net = ptan.agent.TargetNet(net)

print(net.ff.weight)
print(tgt_net.target_model.ff.weight)

net.ff.weight.data += 1.0

tgt_net.sync()

