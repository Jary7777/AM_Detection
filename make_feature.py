#!/usr/bin/env python

# ------------------------------
# -*- coding:utf-8 -*-
# author:JiaLiu
# Email:jaryliu1997@gmail.com
# Datetime:2024/3/28 15:48
# File:make_feature.py
# ------------------------------


import csv
import os
from apk_parse import APK

# APK文件夹路径
apk_folder_path = 'APK'

# 新CSV文件路径
new_csv_file_path = 'new.csv'

# 要提取的特征列表，来自feature.csv的第一行，假设你已经知道这些特征的名称
# 示例：features_to_extract = ['feature1', 'feature2', 'feature3', ...]
features_to_extract = ['feature_name1', 'feature_name2', ...]

# 初始化新CSV文件并写入标题
with open(new_csv_file_path, mode='w', newline='') as new_file:
    csv_writer = csv.writer(new_file)
    csv_writer.writerow(features_to_extract)

# 遍历APK文件夹中的每个APK文件
for apk_file in os.listdir(apk_folder_path):
    if apk_file.endswith('.apk'):
        # APK文件的完整路径
        apk_file_path = os.path.join(apk_folder_path, apk_file)

        # 解析APK文件
        apk = APK(apk_file_path)

        # 提取特征
        extracted_features = []
        for feature in features_to_extract:
            # 假设有一个方法能够从APK对象中获取特征
            extracted_feature = getattr(apk, feature, None)
            extracted_features.append(extracted_feature)

        # 将提取的特征写入新的CSV文件
        with open(new_csv_file_path, mode='a', newline='') as new_file:
            csv_writer = csv.writer(new_file)
            csv_writer.writerow(extracted_features)

print('Feature extraction completed.')
