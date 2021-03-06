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

from .twp_eval_metrics_cls import *
from .twp_NN import *
from .twp_params_table import *

seed = 58
np.random.seed(seed)
torch.manual_seed(seed)

def prep(save, d_feature, lamb, dataset,
         K, val_batch_size, num_neighbors,
         eta_1, eps):
    now = datetime.datetime.now()
    exp_name = now.strftime("/%Y-%m-%d-%H-%M-%S-TWP-" + dataset)
    save_folder = save + exp_name
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    val_save_path = save_folder + r"/val.txt"
    with open(save_folder + r'/hyper-parameters.txt', 'wb') as f:
        t = hyper_params(d_feature, lamb, dataset,
                         K, val_batch_size, num_neighbors,
                         eta_1, eps)
        f.write(t)

    return val_save_path


def twp(d_feature, lamb, tasks, data_path, dataset, save,
        K, val_batch_size, num_neighbors,
        eta_1, eps,num_iterations):
    val_save_path = prep(save, d_feature, lamb, dataset,
                         K, val_batch_size, num_neighbors,
                         eta_1, eps)
    net = NN(d_feature + 1)
    # buffer = pd.DataFrame()
    T = len(tasks)
    res = []
    res_check=[]
    aucs=[]

    for t in range(1, T + 1):
        start_time = time.time()
        criterion = nn.BCELoss()
        task0 = pd.read_csv(data_path + '/' + dataset + r'/task' + str(t) + r'/task' + str(t) + '_neg.csv')
        task1 = pd.read_csv(data_path + '/' + dataset + r'/task' + str(t) + r'/task' + str(t) + '_pos.csv')
        train_task0 = task0.sample(K)
        train_task1 = task1.sample(K)
        task = pd.concat([task0, task1])
        train_task = pd.concat([train_task0, train_task1])

        temp_weights = [w.clone() for w in list(net.parameters())]

        val_batch = round(len(task.index) * val_batch_size)
        val_task = task.sample(val_batch)

        X_q = val_task[val_task.columns[-d_feature:]].copy()
        y_q = val_task[["y"]].values
        z_q = val_task[["z"]]
        X_q = pd.concat([X_q, z_q], axis=1).values
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
        loss = criterion(y_hat, y_q) / len(task.index)

        fair = torch.abs(torch.mean((z_q - z_bar) * y_hat)).item()

        y_hat = y_hat.detach().numpy().reshape(len(y_hat), 1)
        y_q = y_q.detach().numpy().reshape(len(y_q), 1)

        input_zy = np.column_stack((z_temp, y_hat))
        z_y_hat_y = np.column_stack((input_zy, y_temp))
        yX = np.column_stack((y_hat, X_temp))

        accuracy = accuracy_score(y_hat.round(), y_q)
        dp = cal_dp(input_zy)
        eop = cal_eop(z_y_hat_y)
        auc=cla_auc_fairness(input_zy)
        aucs.append(auc)
        discrimination = cal_discrimination(input_zy)
        # consistency = cal_consistency(yX, num_neighbors)
        consistency = 1

        cost_time = time.time() - start_time
        print("Val-Task %s/%s: loss:%s; dbc:%s; acc:%s ;dp:%s; eop:%s; disc:%s; cons:%s; time:%s sec." % (
            t, T, np.round(loss.item(), 4), np.round(fair, 10), np.round(accuracy, 10), np.round(dp, 10), np.round(eop, 10),
            np.round(discrimination, 10), np.round(consistency, 10),
            np.round(cost_time, 4)))
        # torch.save(net.state_dict(), model_save_path)
        res.append(["Val-Task %s/%s: loss:%s; dbc:%s; acc:%s ;dp:%s; eop:%s; disc:%s; cons:%s; time:%s sec." % (
            t, T, np.round(loss.item(), 4), np.round(fair, 10), np.round(accuracy, 10), np.round(dp, 10), np.round(eop, 10),
            np.round(discrimination, 10), np.round(consistency, 10),
            np.round(cost_time, 4))])
        res_check.append(["Val-Task %s/%s: dp:%s; eop:%s." % (t, T, np.round(dp, 10), np.round(eop, 10))])


        # temp_weights = [w.clone() for w in list(net.parameters())]
        X_s = train_task[train_task.columns[-d_feature:]].copy()
        y_s = train_task[["y"]].values
        z_s = train_task[["z"]]
        X_s = pd.concat([X_s, z_s], axis=1).values
        z_s = z_s.values
        z_bar = np.mean(z_s) * np.ones((len(z_s), 1))

        X_s = torch.tensor(X_s, dtype=torch.float).unsqueeze(1)
        y_s = torch.tensor(y_s, dtype=torch.float).unsqueeze(1)
        ones = torch.tensor(np.ones((len(y_s), 1)), dtype=torch.float).unsqueeze(1)
        z_s = torch.tensor(z_s, dtype=torch.float).unsqueeze(1)
        z_bar = torch.tensor(z_bar, dtype=torch.float).unsqueeze(1)

        for iter in range(1, num_iterations + 1):

            y_hat = net.parameterised(X_s, temp_weights)
            fair = torch.abs(torch.mean((z_s - z_bar) * y_hat)) - eps
            loss = (-1.0) * torch.mean(y_s * torch.log(y_hat) + (ones - y_s) * torch.log(ones - y_hat)) + lamb * fair
            loss = loss / len(train_task.index)
            grad = torch.autograd.grad(loss, temp_weights)
            temp_weights = [w - eta_1 * g for w, g in zip(temp_weights, grad)]
            temp_weights_norm = net.e_norm(temp_weights)
            if temp_weights_norm > 100:
                temp_weights = [w / net.e_norm(temp_weights) for w in temp_weights]

            weights = list(nn.parameter.Parameter(item) for item in temp_weights)
            net.assign(weights)




    with open(val_save_path, 'wb') as f:
        pickle.dump(res, f)

    return res, res_check,aucs
