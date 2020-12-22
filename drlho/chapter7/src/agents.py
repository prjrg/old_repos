import torch
import torch.nn as nn
import ptan


class DQNNet(nn.Module):
    def __init__(self, actions: int):
        super(DQNNet, self).__init__()
        self.actions = actions

    def forward(self, x):
        return torch.eye(x.size()[0], self.actions)


net = DQNNet(actions=3)
selector = ptan.actions.ArgmaxActionSelector()
agent = ptan.agent.DQNAgent(dqn_model=net, action_selector=selector)
print(agent(torch.zeros(2,5)))

selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0)
agent = ptan.agent.DQNAgent(dqn_model=net, action_selector=selector)
print(agent(torch.zeros(10, 5))[0])

selector.epsilon = 0.5
print(agent(torch.zeros(10, 5))[0])
selector.epsilon = 0.1
print(agent(torch.zeros(10, 5))[0])


class PolicyNet(nn.Module):
    def __init__(self, actions: int):
        super(PolicyNet, self).__init__()
        self.actions = actions

    def forward(self, x):
        shape = (x.size()[0], self.actions)
        res = torch.zeros(shape, dtype=torch.float32)
        res[:, 0] = 1
        res[:, 1] = 1
        return res


net = PolicyNet(actions=5)
print(net(torch.zeros(6, 10)))
selector = ptan.actions.ProbabilityActionSelector()
agent = ptan.agent.PolicyAgent(model=net, action_selector=selector, apply_softmax=True)
print(agent(torch.zeros(6, 5))[0])



