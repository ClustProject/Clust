import sys
import torch
import numpy as np
import random
sys.path.append("..")
sys.path.append("../..")

from Clust.clust.ML.classification.classification_model.cnn_1d_model import CNNModel
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
        

    def set_param(self, train_params):
        """
        Set Parameter for train

        Args:
            param(dict): parameter for train


        >>> param = { "device":"cpu",
        ...          "batch_size":16,
        ...          "num_classes":,
        ...          "n_epochs":10,}

        """
        self.train_params = train_params


    def set_model(self, model_method, model_params):
        """
        Build model and return initialized model for selected model_name

        Args:
            model_method (string): model method name
        """
       
        self.model_params = model_params

        # build initialized model
        if (model_method == 'LSTM_cf') | (model_method == "GRU_cf"):
            self.model = RNNModel(self.model_params)
        elif model_method == 'CNN_1D_cf':
            self.model = CNNModel(self.model_params)
        elif model_method == 'LSTM_FCNs_cf':
            self.model = LSTMFCNsModel(self.model_params)
        else:
            print('Choose the model correctly')


    def set_data(self, train_x, train_y, val_x, val_y):
        """
        transform data for train

        Args:
            train_x (np.array): train X data
            train_y (np.array): train y data
            val_x (np.array): validation X data
            val_y (np.array): validation y data
        """
        self.train_loader, self.valid_loader = self.model.create_trainloader(self.train_params['batch_size'], train_x, train_y, val_x, val_y)
        
        # self.params['input_size'] = input_size
        # self.params['seq_len'] = seq_len


    def train(self):
        """
        Train and return model

        """
        print("Start training model")
        
        self.model.train(self.train_params, self.train_loader, self.valid_loader)


    def save_best_model(self, save_path):
        """
        Save the best model to save_path

        Args:
            save_path (string): path to save model
        """
        self.model.save_model(save_path)