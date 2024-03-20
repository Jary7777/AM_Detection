#!/usr/bin/env python

# ------------------------------
# -*- coding:utf-8 -*-
# author:JiaLiu
# Email:jaryliu1997@gmail.com
# Datetime:2024/3/20 14:57
# File:dataloader.py
# ------------------------------

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class dataloader():
    def __init__(self, csv_file_path):
        self.filename = csv_file_path
    def load(self):
        # 读csv文件
        df = pd.read_csv(self.filename)
        label_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        df['Class'] = df['Class'].map(label_mapping)
        input_features = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values
        num_features = input_features.shape[1]
        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(input_features)

        # totensor
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        X_train, X_test, y_train, y_test = train_test_split(input_tensor, label_tensor, test_size=0.2, random_state=42)
        print(f'-----train_dataset:{X_train.shape}-----')
        print(f'-----test_dataset:{X_test.shape}-----')

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

        return train_loader, test_loader, num_features, y_test

if __name__ == '__main__':

    print(dataloader(csv_file_path = 'CSV/feature_vectors_syscallsbinders_frequency_5_Cat.csv').load())


