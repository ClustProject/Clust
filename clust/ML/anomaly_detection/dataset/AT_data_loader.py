import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


class CustomSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        # self.scaler = StandardScaler()

        # KETI 방식으로 밖에서 데이터 선언 시 주석 
        train_ver2 = pd.read_csv(data_path + '/train_ver2.csv')
        test_ver2 = pd.read_csv(data_path + '/test_ver2.csv')
        inference_ver2 = pd.read_csv(data_path + '/inference_ver2.csv')

        self.train = train_ver2[:int(len(train_ver2)*0.8)].iloc[:,1:-1].values
        self.valid = train_ver2[int(len(train_ver2)*0.8):].iloc[:,1:-1].values
        self.inference = inference_ver2.iloc[:,1:-1].values

        self.test = test_ver2.values[:, 1:-1]
        self.test_labels = test_ver2.values[:, -1]

        if self.mode == 'train':
             print("train:", self.train.shape)
        elif (self.mode == 'test'):
             print("test:", self.test.shape)
        elif (self.mode == 'inference'):
             print("inference:", self.inference.shape)
        # print("test:", self.test.shape)
        # print("train:", self.train.shape)


    def __len__(self):
            """
            Number of images in the object dataset.
            """
            if self.mode == "train":
                return (self.train.shape[0] - self.win_size) // self.step + 1
            elif (self.mode == 'val'):
                return (self.val.shape[0] - self.win_size) // self.step + 1
            elif (self.mode == 'test'):
                return (self.test.shape[0] - self.win_size) // self.step + 1
            elif (self.mode == 'inference'): # 실질적으로는 inference_threshold_loader를 만들기 위함.
                return (self.inference.shape[0] - self.win_size) // self.win_size + 1
            else:
                return (self.test.shape[0] - self.win_size) // self.win_size + 1


    def __getitem__(self, index):
            index = index * self.step
            if self.mode == "train":
                return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
            elif (self.mode == 'val'):
                return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
            elif (self.mode == 'test'):
                return np.float32(self.test[index:index + self.win_size]), np.float32(
                    self.test_labels[index:index + self.win_size])
            elif (self.mode == 'inference'):
                return np.float32(self.inference[
                                index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                    self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            else:
                return np.float32(self.test[
                                index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                    self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])



def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='Custom'):
    if (dataset == 'Custom'):
        dataset = CustomSegLoader(data_path, win_size, 1, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader