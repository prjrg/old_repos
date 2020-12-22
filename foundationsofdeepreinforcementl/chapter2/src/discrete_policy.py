from torch.distributions import Categorical
import torch

policy_net_output = torch.tensor([-1.6094, -0.2231])

pdparams = policy_net_output
pd = Categorical(logits=pdparams)

action = pd.sample()
pd.log_prob(action)
