#!/usr/bin/env python

# ------------------------------
# -*- coding:utf-8 -*-
# author:JiaLiu
# Email:jaryliu1997@gmail.com
# Datetime:2024/3/20 14:55
# File:train_and_test.py
# ------------------------------

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from datasets.dataloader import dataloader
from network import SimpleCNN, AttentionCNN
from src.args import args
from src.save_and_load_model import save_model, load_model
from tools.Calculation import calculate_metrics, plot_confusion_matrix
from itertools import cycle
from src.csv_file_path import csv_file_path
from sklearn.metrics import classification_report


class train():
    def __init__(self):
        super().__init__()
    print(f'args: {args}')
    train_loader, test_loader, num_features, y_test = dataloader(csv_file_path).load()
    # 模型、定义损失函数和优化器
    model = SimpleCNN(num_features=num_features)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])
    best_accuracy = 0
    losses = []
    num_epochs = args['num_epochs']

    for epoch in range(num_epochs):
        total_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}')
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = correct / total
            print(f'Epoch {epoch + 1}: Test Accuracy: {accuracy}')

            if accuracy > best_accuracy:
                save_model(model)
                best_accuracy = accuracy
            model.train()

    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()

    # test
    model.eval()
    y_true = []
    y_pred = []
    best_model = SimpleCNN(num_features=num_features)
    load_model(best_model)
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = best_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    classes = [0, 1, 2, 3, 4]
    cm, metrics_per_class, macro_avg = calculate_metrics(y_true, y_pred, classes)

    # print("Metrics for each class:")
    # for idx, class_id in enumerate(classes):
    #     print(f"Class {class_id}:")
    #     print(f"  Precision: {metrics_per_class['precision'][idx]}")
    #     print(f"  Recall: {metrics_per_class['recall'][idx]}")
    #     print(f"  F1: {metrics_per_class['f1'][idx]}")
    #
    # print("\nMacro Average Metrics:")
    # print(f"  Precision: {macro_avg['precision']}")
    # print(f"  Recall: {macro_avg['recall']}")
    # print(f"  F1: {macro_avg['f1']}")
    #
    # accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
    # print(f"\nAccuracy: {accuracy}")
    Risk_name = {0: 'Advertising software', 1: 'Bank malware', 2: 'SMS malware', 3: 'Risk software', 4: 'Normal '}
    class_names = list(Risk_name.values())
    print(classification_report(y_true, y_pred, target_names=class_names))
    # ----------------------------------------------------------------
    y_test_binarized = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
    n_classes = y_test_binarized.shape[1]
    print(f"\nNumber of classes: {n_classes}")

    y_scores = torch.zeros((len(y_test), n_classes))

    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            outputs = model(inputs)
            y_scores[i * test_loader.batch_size:(i + 1) * test_loader.batch_size] = outputs

    # ROC、AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_scores[:, i].numpy())
        roc_auc[i] = auc(fpr[i], tpr[i])

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], y_scores[:, i].numpy())
        average_precision[i] = auc(recall[i], precision[i])

    plot_confusion_matrix(cm, class_names=Risk_name.values())
    # ROC曲线
    plt.figure(figsize=(7, 5))
    colors = cycle(['blue', 'red', 'green', 'cyan', 'yellow'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, label=f'{Risk_name[i]} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for each class')
    plt.legend(loc="lower right")
    plt.show()

    # P-R曲线
    plt.figure(figsize=(7, 5))
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color,
                 label=f' {Risk_name[i]} ')  #(area = {average_precision[i]:.2f})
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve for each class')
    plt.legend(loc="lower left")
    plt.show()

if __name__ == '__main__':
    train()