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

    return metrics_per_class, macro_avg