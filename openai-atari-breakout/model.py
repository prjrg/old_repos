import torch
from torch import nn
from torch.nn import Conv2d, LSTM, Embedding, BatchNorm2d
from torch.nn.init import kaiming_normal_

from mish_activation import Mish, mish

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def size(fsize, ksize, padding, strides):
    return int((fsize - ksize + 2*padding)/strides) + 1

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

# TODO
class QNetwork(nn.Module):
    def __init__(self, input_size, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.conv1 = Conv2d(state_size, 32, 3, 2, padding=2)
        ssize = size(input_size, 3, 2, 2)
        self.bn1 = BatchNorm2d(32)
        self.conv2 = Conv2d(32, 64, 3, 2, padding=2)
        ssize = size(ssize, 3, 2, 2)
        self.bn2 = BatchNorm2d(64)
        self.conv3 = Conv2d(64, 128, 3, 2, padding=2)
        ssize = size(ssize, 3, 2, 2)
        self.bn3 = BatchNorm2d(128)
        self.conv4 = Conv2d(128, 256, 1, 1, padding=1)
        ssize = size(ssize, 1, 1, 1)
        self.bn4 = BatchNorm2d(256)

        self.lstm = LSTM(input_size=256*ssize*ssize, hidden_size=512, num_layers=2, dropout=0.3, batch_first=True)

        self.value_function = nn.Sequential(
            nn.Linear(512, 256),
            Mish(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

        self.advantage_function = nn.Sequential(
            nn.Linear(512, 256),
            Mish(),
            nn.Dropout(0.5),
            nn.Linear(256, action_size)
        )

        for m in self.modules():
            weights_init(m)



    def init_hidden(self, batch_size):
        return (torch.zeros(2, batch_size, self.hidden_size).to(device),
                torch.zeros(2, batch_size, self.hidden_size).to(device))

    def forward(self, x, hidden):
        x = mish(self.bn1(self.conv1(x)))
        x = mish(self.bn2(self.conv2(x)))
        x = mish(self.bn3(self.conv3(x)))
        x = mish(self.bn4(self.conv4(x)))

        x = x.reshape(x.size[0], 1, -1)
        x, hidden = self.lstm(x, hidden)
        x = x.contiguous().view(x.size(0), -1)
        vf = self.value_function(x)
        adv = self.advantage_function(x)
        adv = adv - adv.mean(dim=1)

        return vf + adv
