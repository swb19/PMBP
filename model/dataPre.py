# -*- coding: utf-8 -*-
from io import open
import os.path
from os import path
import random
import numpy as np

import pickle
import pandas as pd
import scipy.signal
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import matplotlib.pyplot as plt

# plt.switch_backend('TkAgg')
import matplotlib.ticker as ticker
import numpy as np

BatchSize = 128


class TrajectoryDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, y_label, noise_mode=False, augment=False):

        self.csv_file = csv_file
        self.noise_mode = noise_mode
        self.augment = augment
        self.load_grip_data()
        self.y_label = y_label

        # self.normalize_data()

    def __len__(self):
        return len(self.X_frames_history)

    def traj_augment(self, history_xy, pred_xy, gt_xy):
        if np.random.random() > 0.5:
            angle = 2 * np.pi * np.random.random()
            sin_angle = np.sin(angle)
            cos_angle = np.cos(angle)

            angle_mat = np.array(
                [[cos_angle, -sin_angle],
                 [sin_angle, cos_angle]])

            all_pred_traj = np.vstack([history_xy, pred_xy]).transpose(1, 0)
            all_gt_traj = np.vstack([history_xy, gt_xy]).transpose(1, 0)

            all_pred_traj = np.einsum('ab,bt->at', angle_mat, all_pred_traj)
            all_gt_traj = np.einsum('ab,bt->at', angle_mat, all_gt_traj)

            all_pred_traj = all_pred_traj.transpose(1, 0)
            history_xy = all_pred_traj[:6, :]
            pred_xy = all_pred_traj[6:, :]
            all_gt_traj = all_gt_traj.transpose(1, 0)
            gt_xy = all_gt_traj[6:, :]
        return history_xy, pred_xy, gt_xy

    def __getitem__(self, idx):
        history_xy = self.X_frames_history[idx].astype(np.float64) # 目前是针对
        pred_xy = self.X_frames_pred[idx].astype(np.float64)
        gt_xy = self.X_frames_fut[idx].astype(np.float64)
        gt_ADE = self.ADE_frames[idx].astype(np.float64)
        gt_FDE = self.FDE_frames[idx].astype(np.float64)

        error_xy = pred_xy - gt_xy
        if '_with_scaler' in self.csv_file:
            pred_xy_no_scaler = self.X_frames_pred_no_scaler[idx].astype(np.float64)
            gt_xy_no_scaler = self.X_frames_fut_no_scaler[idx].astype(np.float64)
            error_xy_no_scaler = pred_xy_no_scaler - gt_xy_no_scaler
            dis_error_no_scaler = ((error_xy_no_scaler ** 2).sum(-1)) ** 0.5
            dis_error_with_scale = ((error_xy ** 2).sum(-1)) ** 0.5
        else:
            dis_error_no_scaler = ((error_xy**2).sum(-1))**0.5 # 不支持原始数据下的_with_scaler模式

        # TODO: 数据增强
        if self.augment:
            history_xy, pred_xy, gt_xy = self.traj_augment(history_xy, pred_xy, gt_xy)

        if self.y_label == 'ADE':
            return (history_xy, pred_xy, gt_ADE)
        elif self.y_label == 'FDE':
            return (history_xy, pred_xy, gt_FDE)
        elif self.y_label == 'all_fut':
            return (history_xy, pred_xy, error_xy)
        elif self.y_label == 'dis':
            return (history_xy, pred_xy, dis_error_no_scaler)
        elif 'OOD' in self.y_label:
            return (history_xy, pred_xy, gt_xy)
        elif self.y_label in ['dis_with_ood', 'dis_with_OD']:
            return (history_xy, pred_xy, gt_xy, dis_error_no_scaler)
        elif self.y_label == 'dis_with_scaler':
            return (history_xy, pred_xy, dis_error_with_scale)
        elif self.y_label in ['dis_with_scaler_with_ood', 'dis_with_scaler_with_OD']:
            return (history_xy, pred_xy, gt_xy, dis_error_with_scale)
        # elif self.y_label == 'OOD_history':
        #     return (history_xy, pred_xy, gt_ADE)
        # elif self.y_label == 'OOD_fut':
        #     return (gt_xy, pred_xy, gt_ADE)
        # elif self.y_label == 'OOD_all':
        #     return (np.vstack([history_xy, gt_xy]), pred_xy, gt_ADE)


    def load_grip_data(self):
        with open(self.csv_file, 'rb') as file:
            dataS = pickle.load(file)
        # 过滤真实轨迹
        # if 'gt_len' in dataS.columns:
        #     dataS = dataS[(dataS['gt_len']==6)&(dataS['history_len']==6)]
        # TODO： 针对noise文件提取包含noise
        if self.noise_mode:
            self.X_frames_history = dataS['history_xy_noise'].values
            if '_with_scaler' in self.csv_file:
                self.X_frames_history_no_scaler = dataS['history_xy_noise_no_scaler'].values
        else:
            self.X_frames_history = dataS['history_xy'].values
            if '_with_scaler' in self.csv_file:
                self.X_frames_history_no_scaler = dataS['history_xy_no_scaler'].values
        self.X_frames_pred = dataS['pred_xy'].values
        self.X_frames_fut = dataS['gt_xy'].values
        if '_with_scaler' in self.csv_file:
            self.X_frames_pred_no_scaler = dataS['pred_xy_no_scaler'].values
            self.X_frames_fut_no_scaler = dataS['gt_xy_no_scaler'].values
        self.ADE_frames = dataS['ADE'].values
        self.FDE_frames = dataS['FDE'].values



    # TODO: 可以尝试归一化？
    # def normalize_data(self):
    #     A = [list(x) for x in zip(*(self.X_frames))]
    #     A = torch.tensor(A)
    #     A = A.view(-1, A.shape[2])
    #     print('A:', A.shape)
    #     self.mn = torch.mean(A, dim=0)
    #     self.range = (torch.max(A, dim=0).values - torch.min(A, dim=0).values) / 2.0
    #     self.range = torch.ones(self.range.shape, dtype=torch.double)
    #     self.std = torch.std(A, dim=0)
    #     # self.X_frames = [torch.tensor(item) for item in self.X_frames]
    #     # self.Y_frames = [torch.tensor(item) for item in self.Y_frames]
    #     self.X_frames = [(torch.tensor(item) - self.mn) / (self.std * self.range) for item in self.X_frames]
    #     self.Y_frames = [(torch.tensor(item) - self.mn[:4]) / (self.std[:4] * self.range[:4]) for item in self.Y_frames]


def get_dataloader(csv_file, y_label='ADE', noise_mode=False, augment=False):
    '''
    return torch.util.data.Dataloader for train,test and validation
    '''
    # load dataset
    if path.exists("my_dataset.pickle"):
        with open('my_dataset.pickle', 'rb') as data:
            dataset = pickle.load(data)
    else:
        dataset = TrajectoryDataset(csv_file, y_label, noise_mode=noise_mode, augment=augment)
        # with open('my_dataset.pickle', 'wb') as output:
        #     pickle.dump(dataset, output)
    # split dataset into train test and validation 7:2:1
    num_train = (int)(dataset.__len__() * 0.7)
    num_test = (int)(dataset.__len__() * 0.9) - num_train
    num_validation = (int)(dataset.__len__() - num_test - num_train)
    train, test, validation = torch.utils.data.random_split(dataset, [num_train, num_test, num_validation])
    train_loader = DataLoader(train, batch_size=BatchSize, shuffle=True)
    test_loader = DataLoader(test, batch_size=BatchSize, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=BatchSize, shuffle=True)
    return (train_loader, test_loader, validation_loader, dataset)


if __name__ == '__main__':
    Training_generator, Test, Valid, WholeSet = get_dataloader()
    # for local_batch, local_labels in Training_generator:
    #     print(local_batch)
    for history_xy, pred_xy, gt_ADE in Training_generator:
        print(history_xy)
        print(gt_ADE)