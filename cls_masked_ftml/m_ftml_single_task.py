from __future__ import division
import torch, os
import torch.nn as nn
import random
import math
import copy, time, pickle, datetime
import pandas as pd
import numpy as np
from numpy import linalg as LA
from sklearn.metrics import accuracy_score


from .m_ftml_eval_metrics_cls import *
from .m_ftml_NN import *


def sample_data_from_task(task, batch_size, d_feature):
    data = task.sample(batch_size)
    X = data[data.columns[-d_feature:]].copy()
    y = data[["y"]]
    z = data[["z"]]
    return X, y, z


def mean(a):
    return sum(a).to(dtype=torch.float) / len(a)


def cal_loss_and_fairness(d_feature, net, task,
                          K, Kq, num_neighbors,
                          inner_steps,
                          eta_1):
    # if t != 1:
    #     net = NN(d_feature)
    #     net.load_state_dict(torch.load(model_save_path))
    # else:
    #     net = net
    temp_weights = [w.clone() for w in list(net.parameters())]

    criterion = nn.BCELoss()

    task0 = task[0]
    task1 = task[1]
    task_df = pd.concat([task0, task1])

    X0_s, y0_s, z0_s = sample_data_from_task(task0, K, d_feature)
    X1_s, y1_s, z1_s = sample_data_from_task(task1, K, d_feature)
    X_s = pd.concat([X0_s, X1_s]).values
    y_s = pd.concat([y0_s, y1_s]).values

    X_s = torch.tensor(X_s, dtype=torch.float).unsqueeze(1)
    y_s = torch.tensor(y_s, dtype=torch.float).unsqueeze(1)

    for step in range(inner_steps):
        loss = criterion(net.parameterised(X_s, temp_weights), y_s) / K
        grad = torch.autograd.grad(loss, temp_weights)
        temp_weights = [w - eta_1 * g for w, g in zip(temp_weights, grad)]

    if Kq < 1:
        Kq = round(len(task_df.index) * Kq)

    X_q, y_q, z_q = sample_data_from_task(task_df, Kq, d_feature)
    X_q = X_q.values
    y_q = y_q.values
    z_q = z_q.values

    X_temp = copy.deepcopy(X_q)
    z_temp = copy.deepcopy(z_q)
    y_temp = copy.deepcopy(y_q)
    z_bar = np.mean(z_q) * np.ones((len(z_q), 1))

    X_q = torch.tensor(X_q, dtype=torch.float).unsqueeze(1)
    y_q = torch.tensor(y_q, dtype=torch.float).unsqueeze(1)
    z_q = torch.tensor(z_q, dtype=torch.float).unsqueeze(1)
    z_bar = torch.tensor(z_bar, dtype=torch.float).unsqueeze(1)

    y_hat = net.parameterised(X_q, temp_weights)
    loss = criterion(y_hat, y_q)
    loss = loss / Kq

    fair = torch.abs(torch.mean((z_q - z_bar) * y_hat)).item()

    y_hat = y_hat.detach().numpy().reshape(len(y_hat), 1)
    y_q = y_q.detach().numpy().reshape(len(y_q), 1)

    input_zy = np.column_stack((z_temp, y_hat))
    z_y_hat_y = np.column_stack((input_zy, y_temp))
    # yX = np.column_stack((y_hat, X_temp))

    accuracy = accuracy_score(y_hat.round(), y_q)
    auc = cla_auc_fairness(input_zy)
    dp = cal_dp(input_zy)
    eop = cal_eop(z_y_hat_y)
    discrimination = cal_discrimination(input_zy)
    consistency = 1

    return loss, fair, accuracy, dp, eop, discrimination, consistency,auc
