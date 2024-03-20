#!/usr/bin/env python

# ------------------------------
# -*- coding:utf-8 -*-
# author:JiaLiu
# Email:jaryliu1997@gmail.com
# Datetime:2024/3/20 14:50
# File:save_and_load_model.py
# ------------------------------
import torch

def save_model(model, path="breast_weights/best_model.pth"):
    torch.save(model.state_dict(), path)


def load_model(model, path="breast_weights/best_model.pth"):
    model.load_state_dict(torch.load(path))
    model.eval()
