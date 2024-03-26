#!/usr/bin/env python

# ------------------------------
# -*- coding:utf-8 -*-
# author:JiaLiu
# Email:jaryliu1997@gmail.com
# Datetime:2024/3/25 15:02
# File:html_display.py
# ------------------------------

from flask import Flask, render_template, send_from_directory

app = Flask(__name__)

@app.route('/')
def index():
    with open('result/classification_report.txt', 'r') as file:
        report = file.read()

    return render_template('index.html', report=report)

@app.route('/results/<path:filename>')
def results(filename):
    return send_from_directory('result', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5004)
