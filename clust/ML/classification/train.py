import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import copy
import datetime
from torch.utils.data import TensorDataset, DataLoader
sys.path.append("..")
sys.path.append("../..")

from Clust.clust.transformation.type.DFToNPArray import transDFtoNP
from Clust.clust.ML.classification.classification_model.cnn_1d_model import CNNModel
from Clust.clust.ML.classification.classification_model.fc_model import FCModel
from Clust.clust.ML.classification.classification_model.lstm_fcns_model import LSTMFCNsModel
from Clust.clust.ML.classification.classification_model.rnn_model import RNNModel

class ClassificationTrain():
    def __init__(self):
        # seed 고정
        random_seed = 42

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)
        # super().__init__()
        

    def set_param(self, params):
        """
        Set Parameter for train

        Args:
        param(dict): parameter for train
            >>> param = { "device":"cpu",
                         "batch_size":16,
                         "num_classes":,
                         "n_epochs":10,}
        """
        self.params = params
        self.n_epochs = params['n_epochs']
        self.num_classes = params['num_classes']
        self.device = params['device']
        self.batch_size = params['batch_size']


    def set_model(self, model_method):
        """
        Build model and return initialized model for selected model_name

        Args:
            model_method (string): model method name
        """
        if model_method == 'LSTM_cf':
            self.params["rnn_type"] = 'lstm'
        elif self.model_method == 'GRU_cf':
            self.params["rnn_type"] = 'gru'
        
        # build initialized model
        if (model_method == 'LSTM_cf') | (model_method == "GRU_cf"):
            self.model = RNNModel(self.params)
        elif model_method == 'CNN_1D_cf':
            self.model = CNNModel(self.params)
        elif model_method == 'LSTM_FCNs_cf':
            self.model = LSTMFCNsModel(self.params)
        elif model_method == 'FC_cf':
            self.model = FCModel(self.params)
        else:
            print('Choose the model correctly')


    def set_data(self, train_x, train_y, val_x, val_y, window_num=0, dim=None):
        """
        transform data for train

        Args:
            train_x (dataframe): train X data
            train_y (dataframe): train y data
            val_x (dataframe): validation X data
            val_y (dataframe): validation y data
            window_num (integer) : window size
        """
        self.train_loader, self.valid_loader = self.model.create_trainloader(self.batch_size, train_x, train_y, val_x, val_y, window_num, dim)
        
        # self.params['input_size'] = input_size
        # self.params['seq_len'] = seq_len


    def train(self):
        """
        Train and return model

        Returns:
            model: train model
        """
        print("Start training model")
        
        self.model.train(self.params, self.train_loader, self.valid_loader, self.n_epochs, self.device)


    def save_best_model(self, save_path):
        """
        Save the best model to save_path

        Args:
            save_path (string): path to save model
        """
        self.model.save_model(save_path)