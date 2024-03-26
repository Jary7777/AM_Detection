#!/usr/bin/env python

# ------------------------------
# -*- coding:utf-8 -*-
# author:JiaLiu
# Email:jaryliu1997@gmail.com
# Datetime:2024/3/26 14:29
# File:upload.py
# ------------------------------

import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from network import SimpleCNN, AttentionClassifier
from src.save_and_load_model import save_model, load_model
app = Flask(__name__)
import joblib

# 加载模型
clf = joblib.load('best_model.joblib')

# model = AttentionClassifier(num_features=470,num_classes=5)
# load_model(model)
# model.eval()


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        file.save(filename)

        # 读取CSV文件并进行预测
        data = pd.read_csv(filename)
        results = []
        class_names = {
            0: 'Advertising software',
            1: 'Bank malware',
            2: 'SMS malware',
            3: 'Risk software',
            4: 'Normal'
        }
        label_names = {1: 'Advertising software', 2: 'Bank malware', 3: 'SMS malware', 4: 'Risk software', 5: 'Normal'}
        #----------------------------------------------------------------
        X = data.drop('Class', axis=1).values
        y_pred = clf.predict(X)
        for index, (prediction, actual_label) in enumerate(zip(y_pred, data['Class'])):
            predicted_class = class_names[prediction]
            actual_class = label_names[actual_label]
            results.append((index, predicted_class, actual_class))

        return render_template('results.html', results=results)
        #----------------------------------------------------------------
        # for index, row in data.iterrows():
        #     features = torch.tensor(row.drop('Class').values.astype(float)).float().unsqueeze(0)
        #     output = model(features)[0]
        #     prediction = output.argmax(1).item()
        #     predicted_class = class_names[prediction]
        #     actual_label = label_names[row['Class'] ]# 获取实际标签
        #     results.append((index, predicted_class, actual_label))
        #
        # return render_template('results.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)
