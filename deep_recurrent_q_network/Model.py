import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from mish import mish


class SkipBlock(nn.Module):
    def __init__(self, feat_inp):
        super(SkipBlock, self).__init__()
        self.feat_inp = feat_inp
        self.conv1 = nn.Conv2d(feat_inp, feat_inp, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(feat_inp, feat_inp, kernel_size=3, stride=1, padding=2)
        self.conv3 = nn.Conv2d(feat_inp, feat_inp, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(feat_inp)
        self.bn2 = nn.BatchNorm2d(feat_inp)
        self.bn3 = nn.BatchNorm2d(feat_inp)

        self.convTrans1 = nn.Conv2d(feat_inp, feat_inp, kernel_size=3, stride=1, padding=2)
        self.convTrans2 = nn.Conv2d(feat_inp, feat_inp, kernel_size=3, stride=1, padding=2)
        self.convTrans3 = nn.Conv2d(feat_inp, feat_inp, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        inp = self.convTrans3(self.convTrans2(self.convTrans1(x)))

        x = mish(self.bn1(self.conv1(x)))
        x = mish(self.bn2(self.conv2(x)))
        x = mish(self.bn3(self.conv3(x)))

        return x + inp


class DeepRecurrentQN(nn.Module):
    def __init__(self, action_size):
        super(DeepRecurrentQN, self).__init__()
        self.action_size = action_size

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

        self.blk1 = SkipBlock(64)
        self.blk2 = SkipBlock(64)

        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, dropout=0.4)
        self.fc1 = nn.Linear(128, 128)
        self.advantage = nn.Linear(128, action_size)
        self.value_func = nn.Linear(128, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x, hidden=None):
        x = mish(self.bn1(self.conv1(x)))
        x = mish(self.bn2(self.conv2(x)))
        x = mish(self.bn3(self.conv3(x)))
        x = self.blk1(x)
        x = self.blk2(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.reshape(1, 1, -1)
        x, hidden = self.lstm(x, hidden)
        x = F.dropout(mish(self.fc1(x)), 0.4)
        advantage = self.advantage(x)
        return (advantage - advantage.mean(1)) + self.value_func(x), hidden

    @staticmethod
    def hidden_initialization(device):
        return (Variable(torch.zeros(2, 1, 128).float()).to(device), Variable(torch.zeros(2, 1, 128).float()).to(device))



