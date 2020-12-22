from collections import OrderedDict

import torch
import torchvision
from torch import nn
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

DENSE_NET201 = torchvision.models.densenet201(pretrained=True, drop_rate=0.5)

for param in DENSE_NET201.parameters():
    param.requires_grad = False


class TrainedDensenet201(nn.Module):
    def __init__(self):
        super(TrainedDensenet201, self).__init__()
        features = DENSE_NET201.features
        self.features = nn.Sequential(
            features.conv0,
            features.norm0,
            features.relu0,
            features.pool0
        )
        self.block1 = features[self.get_index(features, "denseblock1")]
        self.tr1 = features[self.get_index(features, "transition1")]
        self.block2 = features[self.get_index(features, "denseblock2")]
        self.tr2 = features[self.get_index(features, "transition2")]
        self.block3 = features[self.get_index(features, "denseblock3")]
        self.tr3 = features[self.get_index(features, "transition3")]
        self.block4 = features[self.get_index(features, "denseblock4")]

        self.fpn = torchvision.ops.FeaturePyramidNetwork([256, 512, 1792, 1920], 256, extra_blocks=LastLevelMaxPool())

        for inner_block in self.fpn.inner_blocks.children():
            torch.nn.init.xavier_normal_(inner_block.weight)
        for outter_block in self.fpn.layer_blocks.children():
            torch.nn.init.xavier_normal_(outter_block.weight)

        for param in self.fpn.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x1 = self.block1(x)
        x = self.tr1(x1)
        x2 = self.block2(x)
        x = self.tr2(x2)
        x3 = self.block3(x)
        x = self.tr3(x3)
        x4 = self.block4(x)

        y = OrderedDict()
        y['a'] = x1
        y['b'] = x2
        y['c'] = x3
        y['d'] = x4

        return self.fpn(y)

    def get_index(self, features, name):
        return list(dict(features.named_children()).keys()).index(name)


