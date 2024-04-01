#!/usr/bin/env python
# ------------------------------
# -*- coding:utf-8 -*-
# author:JiaLiu
# Email:jaryliu1997@gmail.com
# Datetime:2024/3/25 10:14
# File:Machine Learning Algorithms.py
# ------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from tools.Calculation import plot_confusion_matrix
from src.csv_file_path import csv_file_path
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from itertools import cycle
from sklearn.metrics import classification_report


def main():
    # 读取数据集
    df = pd.read_csv(csv_file_path)

    label_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    df['Class'] = df['Class'].map(label_mapping)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    # 01. 机器学习算法——随机森林分类
    #clf = RandomForestClassifier(n_estimators=200, random_state=42)
    # ----------------------------------------------------------------

    # 02. 机器学习算法——Support Vector Machines, SVM
    #clf = SVC(kernel='linear', C=1.0, random_state=42, probability=True)
    # ----------------------------------------------------------------

    # 03. 机器学习算法——朴素贝叶斯 Naive Bayes
    #clf = GaussianNB()
    # ----------------------------------------------------------------

    # 04. 机器学习算法--K最近相邻
    clf = KNeighborsClassifier(n_neighbors=3)
    # ----------------------------------------------------------------

    # 05. 神经网络--多层感知器
    #clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42)
    clf.fit(X_train, y_train)
    # ----------------------------------------------------------------
    # 测试
    Risk_name = {0: 'Advertising software', 1: 'Bank malware', 2: 'SMS malware', 3: 'Risk software', 4: 'Normal '}
    y_pred = clf.predict(X_test)
    class_names = list(Risk_name.values())
    print(classification_report(y_test, y_pred, target_names=class_names))
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    joblib.dump(clf, 'best_model_GaussianNB.joblib')
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names=Risk_name.values())

    y_score = clf.predict_proba(X_test)

    # ----------------------------------------------------------------
    y_test_binarized = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    n_classes = y_test_binarized.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test_binarized[:, i], y_score[:, i])

    plt.figure()
    lw = 2  # line width
    colors = cycle(['blue', 'red', 'green', 'cyan', 'yellow'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=f'{Risk_name[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    plt.figure()
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=lw,
                 label=f' {Risk_name[i]} ')  #(area = {average_precision[i]:.2f})

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Multi-class Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()
if __name__ == '__main__':
    main()
