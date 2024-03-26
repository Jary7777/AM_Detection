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

class Attention(nn.Module):
    """ 注意力机制模块 """
    def __init__(self, in_features):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(in_features, 1)

    def forward(self, x):
        weights = F.softmax(self.attention_weights(x), dim=1)
        weighted_input = x * weights.expand_as(x)
        return weighted_input, weights

class AttentionClassifier(nn.Module):
    """ 带注意力机制的分类器 """
    def __init__(self, num_features, num_classes):
        super(AttentionClassifier, self).__init__()
        self.fc1 = nn.Linear(num_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.attention = Attention(128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        weighted_x, attention_weights = self.attention(x)
        out = self.fc3(weighted_x)
        return out, attention_weights

# 创建模型实例并测试
num_features = 470  # 例如，假设我们有 470 个特征
num_classes = 5     # 假设我们有 5 个分类
model = AttentionClassifier(num_features, num_classes)

# 假设输入的形状是 [batch_size, num_features]
inputs = torch.rand(32, num_features)  # 32 是批量大小
outputs, attention_weights = model(inputs)

print("Outputs shape:", outputs.shape)  # 应该是 [32, 5]
print("Attention weights shape:", attention_weights.shape)  # 应该是 [32, 128]
