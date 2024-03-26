#!/usr/bin/env python

# ------------------------------
# -*- coding:utf-8 -*-
# author:JiaLiu
# Email:jaryliu1997@gmail.com
# Datetime:2024/3/20 14:52
# File:Calculation.py
# ------------------------------
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import itertools

def calculate_metrics(y_true, y_pred, classes):
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=classes)

    # 计算每个类别的指标
    metrics_per_class = {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    # 计算宏平均（未加权平均）
    macro_avg = {
        'precision': np.mean(precision),
        'recall': np.mean(recall),
        'f1': np.mean(f1)
    }

    return cm, metrics_per_class, macro_avg


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)


    ax.set_title(title, pad=20, fontweight='bold', fontsize=16)
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=12)

    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center", color=color, fontweight="bold", fontsize=14)

    ax.set_ylabel('True label', fontsize=14, labelpad=10)
    ax.set_xlabel('Predicted label', fontsize=14, labelpad=10)
    ax.grid(False)

    plt.tight_layout()

    # plt.show()  # 直接显示

    return fig
