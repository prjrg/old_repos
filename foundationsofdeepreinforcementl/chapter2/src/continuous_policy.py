from torch.distributions import Normal
import torch

policy_net_output = torch.tensor([1.0, 0.2])
pdparams = policy_net_output
pd = Normal(loc=pdparams[0], scale=pdparams[1])
action = pd.sample()
pd.log_prob(action)