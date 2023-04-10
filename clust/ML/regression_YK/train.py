import argparse
import torch
import numpy as np
import random

from Clust.clust.ML.regression_YK.clust_models.rnn_clust import RNNClust
from Clust.clust.ML.regression_YK.clust_models.cnn1d_clust import CNN1DClust
from Clust.clust.ML.regression_YK.clust_models.lstm_fcns_clust import LSTMFCNsClust
from Clust.clust.ML.regression_YK.clust_models.fc_clust import FCClust

# seed 고정
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


class RegressionTrain():
    def __init__(self):
        pass

    def set_param(self, train_params):
        """
        Set Parameters for train

        Args:
        params(dict): parameters for train

        Example:

            >>> params = { 'num_layers': 2, 
            ...            'hidden_size': 64, 
            ...            'dropout': 0.1,
            ...            'bidirectional': True,
            ...            'lr':0.0001,
            ...            'device':"cpu",
            ...            'batch_size':16,
            ...            'n_epochs':10    }
        """
        # TODO: parameters refactoring 
        self.train_params = train_params
        # self.batch_size = params['batch_size']
        # self.n_epochs = params['n_epochs']
        # self.device = params['device']

    def set_model(self, model_method, model_params):
        """
        Set model for selected model_method

        Args:
            model_method (string): model method name  
        """
        self.model_params = model_params

        if model_method == 'LSTM_rg' or model_method == 'GRU_rg' or model_method == 'RNN_rg':
            self.model = RNNClust(self.model_params)
        elif model_method == 'CNN_1D_rg':
            self.model = CNN1DClust(self.model_params)
        elif model_method == 'LSTM_FCNs_rg':
            self.model = LSTMFCNsClust(self.model_params)
        elif model_method == 'FC_rg':
            self.model = FCClust(self.model_params)
        else:
            print('Choose the model correctly')

    def set_data(self, train_x, train_y, val_x, val_y):
        """
        set train, val data & transform data for training

        # regression task
        Args:
            train_x (dataframe): train X data
            train_y (dataframe): train y data (optional)
            val_x (dataframe): validation X data
            val_y (dataframe): validation y data (optional)
        """
        self.train_loader, self.valid_loader = self.model.create_trainloader(self.train_params['batch_size'], train_x, train_y, val_x, val_y)

    def train(self):
        """
        Train the model
        """
        print("Start training model")

        # train model
        self.model.train(self.train_params, self.train_loader, self.valid_loader)

    def save_best_model(self, save_path):
        """
        Save the best model to save_path

        Args:
            save_path (string): path to save model
        """
        self.model.save_model(save_path)


# TODO: train 모듈 별도 실행 필요할 경우?
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    
    return parser.parse_known_args()[0] if known else parser.parse_args()

def main():
    pass

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)