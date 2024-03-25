#!/usr/bin/env python

# ------------------------------
# -*- coding:utf-8 -*-
# author:JiaLiu
# Email:jaryliu1997@gmail.com
# Datetime:2024/3/25 11:06
# File:AttentionCNN.py
# ------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import time


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.bias = bias
        self.attention = nn.Linear(feature_dim, 1, bias=bias)

    def forward(self, x):
        eij = self.attention(x)

        # 计算注意力权重
        eij = torch.tanh(eij)
        a = torch.exp(eij)

        # 归一化注意力权重
        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * a
        return torch.sum(weighted_input, 1)


class AttentionCNN(nn.Module):
    def __init__(self, num_features):
        super(AttentionCNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 5)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.attention = Attention(feature_dim=64, step_dim=128)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))

        # x = self.attention(x)
        #
        # x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x


if __name__ == '__main__':
    start_time = time.time()
    model = AttentionCNN(num_features=470)
    summary(model=model, input_size=(4000,470), batch_size=32, device="cpu")  # N-C-D-H-W
    print("--- %s seconds ---" % (time.time() - start_time))