import torch.nn as nn


class TwoHeadNet(nn.Module):
    def __init__(self, net1, net2, device):
        super(TwoHeadNet, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x, hidden=None, b=True):
        if b:
            return self.net1(x, hidden)
        return self.net2(x, hidden)