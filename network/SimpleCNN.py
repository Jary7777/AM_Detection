#!/usr/bin/env python

# ------------------------------
# -*- coding:utf-8 -*-
# author:JiaLiu
# Email:jaryliu1997@gmail.com
# Datetime:2024/3/20 14:45
# File:SimpleCNN.py
# ------------------------------

import torch.nn as nn
from torchsummary import summary
import time

class SimpleCNN(nn.Module):
    def __init__(self, num_features):
        super(SimpleCNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    start_time = time.time()
    model = SimpleCNN(num_features=470)
    summary(model=model, input_size=(4000,470), batch_size=32, device="cpu")  # N-C-D-H-W
    print("--- %s seconds ---" % (time.time() - start_time))
