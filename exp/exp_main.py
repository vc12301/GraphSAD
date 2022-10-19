from data_provider.data_factory import data_provider
from exp.exp_base import Exp_Base
from models import MCAD, Adaptive_MCAD, MCAD_with_variable_length

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from sklearn.metrics import roc_auc_score

import os
import time

import warnings

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Base):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'MCAD': MCAD,
            'Adaptive_MCAD': Adaptive_MCAD,
            'MCAD_with_variable_length': MCAD_with_variable_length
        }

        model = model_dict[self.args.model].Model(self.args, self.device, step_len=self.args.step_len, num_nodes=self.num_nodes)
        return model

    def _get_data(self):
        semantic_data, temporal_data, train_mask, test_mask, all_mask, stride, num_nodes = data_provider(self.args)
        semantic_data = semantic_data.to(self.device) 
        temporal_data = temporal_data.to(self.device)
        x, y = semantic_data.x, semantic_data.y
        self.num_nodes = num_nodes
        return semantic_data, temporal_data, x, y, train_mask, test_mask, all_mask, stride

    def run(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        model_path = path + '/' + 'checkpoint.pth'
        
        # load data 
        semantic_data, temporal_data, x, y, train_mask, test_mask, all_mask, sliding_wdw = self.data
        pseudo_y = y.clone()
        pseudo_y[test_mask == 1] = 1.
        weights = torch.ones(pseudo_y.shape).to(self.device)
        weights[test_mask == 1] = 0.02
        weights = weights * (all_mask.sum() / weights.sum()) 
        # training
        if self.args.is_train == True:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
            criterion = nn.BCELoss(reduction='none')
            for epoch in range(self.args.train_epochs):
                self.model.train()

                self.model.update_model_params()
                model_optim.zero_grad()
                model_optim.lr = self.args.learning_rate
                epoch_time = time.time()
                logit_semantic, logit_temporal, length_penalty, decode_loss, _ = self.model(semantic_data, temporal_data)
                prob = torch.exp(-(logit_semantic + 0.1 * logit_temporal).square().sum(dim=1))
                losses = criterion(prob[all_mask == 1], pseudo_y[all_mask == 1])
                loss = torch.mean(weights * losses)
                total_loss = loss + self.args.lamb * decode_loss
                total_loss.backward()
                model_optim.step()

                self.model.update_length()
                model_optim.zero_grad()
                model_optim.lr = 5e-4
                epoch_time = time.time()
                logit_semantic, logit_temporal, length_penalty, decode_loss, _ = self.model(semantic_data, temporal_data)
                prob = torch.exp(-(logit_semantic + 0.1 * logit_temporal).square().sum(dim=1))
                losses = criterion(prob[all_mask == 1], pseudo_y[all_mask == 1])
                loss = torch.mean(weights * losses)
                total_loss = loss + length_penalty
                total_loss.backward()
                model_optim.step()

                # model_optim.zero_grad()
                # epoch_time = time.time()
                # logit_semantic, logit_temporal, length_penalty, decode_loss, _ = self.model(semantic_data, temporal_data)
                # prob = torch.exp(-(logit_semantic).square().sum(dim=1))
                # loss = criterion(prob[train_mask == 1], semantic_data.y[train_mask == 1])
                # total_loss = loss + length_penalty + decode_loss
                # total_loss.backward()
                # model_optim.step()


                with torch.no_grad():
                    self.model.eval()
                    logit_semantic, logit_temporal, length_penalty, decode_loss, _ = self.model(semantic_data, temporal_data)
                    prob = torch.exp(-(logit_semantic + 0.1 * logit_temporal).square().sum(dim=1))
                    score = roc_auc_score((1. - y[test_mask == 1]).cpu(), (1. - prob[test_mask == 1]).cpu())
                    true = (1. - semantic_data.y[test_mask == 1][:-4]).cpu()
                    pred = (1. - prob[test_mask == 1][:-4]).cpu()
                    acc_1 = multi_acck(true, pred, 1)
                    acc_3 = multi_acck(true, pred, 3)
                    acc_5 = multi_acck(true, pred, 5)
                    acc_10 = multi_acck(true, pred, 10)
                    print("Epoch: {} cost time: {} AUC: {:.4f}".format(
                        epoch + 1, 
                        time.time() - epoch_time,
                        score
                        )
                    )
                    print(f"Recall: {acc_1}, {acc_3}, {acc_5}, {acc_10}")
                    print('loss:{}, decode loss:{}, length penalty:{}\n'.format(loss, self.args.lamb * decode_loss, length_penalty))

            # self.model.load_state_dict(torch.load(model_path))

        # if not training a new model, load the saved model
        if self.args.is_train == False:
            load_dict = torch.load(model_path)
            net.load_state_dict(load_dict['model_state_dict'])

        # testing
        with torch.no_grad():
            self.model.eval()
            logit_semantic, logit_temporal, length_penalty, decode_loss, _ = self.model(semantic_data, temporal_data)
            prob = torch.exp(-(logit_semantic + 0.1 * logit_temporal).square().sum(dim=1))
            score = roc_auc_score((1. - semantic_data.y[test_mask == 1]).cpu(), (1. - prob[test_mask == 1]).cpu())
            true = (1. - semantic_data.y[test_mask == 1][:-4]).cpu()
            pred = (1. - prob[test_mask == 1][:-4]).cpu()
            acc_1 = multi_acck(true, pred, 1)
            acc_3 = multi_acck(true, pred, 3)
            acc_5 = multi_acck(true, pred, 5)
            acc_10 = multi_acck(true, pred, 10)
            print("AUC:{:.4f}".format(score))
            print(f"Recall: {acc_1}, {acc_3}, {acc_5}, {acc_10}")
            # print(prob[test_mask == 1])

        return self.model, prob[test_mask == 1].cpu(), score, acc_1, acc_3, acc_5, acc_10



def multi_acck(truth,pred,k,tol = 4):
    truth_idx = np.where(truth == 1)[0]
    num_ano = sum((truth_idx[1:] - truth_idx[:-1])>1)+1
    
    ti_s = truth_idx[[True]+list(truth_idx[1:]-truth_idx[:-1]>1)]
    ti_e = truth_idx[list(truth_idx[1:]-truth_idx[:-1]>1)+[True]]
    ano_dict ={}
    
    topk = 0
    sort_idx = list(np.argsort(pred))
    count = 0
    while topk < k*num_ano:
        if len(sort_idx)<1:
            break
        if len(set(np.arange(sort_idx[-1]-tol,sort_idx[-1]+tol+1)).intersection(set( truth_idx)))>0:
            count += 1
            inset = list(set(np.arange(sort_idx[-1]-tol,sort_idx[-1]+tol+1)).intersection(set( truth_idx)))
            for i in range(num_ano):
                if inset[0] >= ti_s[i] and inset[0]<= ti_e[i]:
                    if i not in ano_dict:
                        ano_dict[i] = 1
                        topk +=1
        else:
            topk +=1
        sort_idx.remove(sort_idx[-1])
    return len(ano_dict)/num_ano