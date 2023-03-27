import argparse
import torch
import numpy as np
import random

# from Clust.clust.ML.regression_YK.clust_models.rnn_clust import RNNClust
# from Clust.clust.ML.regression_YK.clust_models.cnn1d_clust import CNN1DClust
# from Clust.clust.ML.regression_YK.clust_models.lstm_fcns_clust import LSTMFCNsClust
# from Clust.clust.ML.regression_YK.clust_models.fc_clust import FCClust
from Clust.clust.ML.regression_YK.clust_models.rnn_forecast import RNNForecast

# seed 고정
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


class ForecastTrain():
    def __init__(self):
        pass

    def set_param(self, params):
        """
        Set Parameters for transform, train

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
        self.params = params
        self.batch_size = params['batch_size']
        self.n_epochs = params['n_epochs']
        self.clean_param = params['clean_param']
        self.train_parameter = params['train_parameter']
        self.transform_parameter = params['transform_parameter']
        self.device = params['device']

    def set_model(self, model_method):
        """
        Set model for selected model_method

        Args:
            model_method (string): model method name  
        """
        if model_method == 'rnn':
            self.params['rnn_type'] = 'rnn'
        elif model_method == 'lstm':
            self.params['rnn_type'] = 'lstm'
        elif model_method == 'gru':
            self.params['rnn_type'] = 'gru'
        else:
            print('Choose the model correctly')
        
        self.model = RNNForecast(self.params)

    def set_data(self, train, val):
        """
        set train, val data & transform data for training

        Args:
            train_x (dataframe): train X data
            train_y (dataframe): train y data
            val_x (dataframe): validation X data
            val_y (dataframe): validation y data
            window_num (integer) : window size
        """

        self.train_loader, self.valid_loader = self.model.create_trainloader(self.batch_size, train, val)

    def train(self):
        """
        Train the model
        """
        print("Start training model")

        # train model
        self.model.train(self.params, self.train_loader, self.valid_loader, self.n_epochs, self.device)

    def export(self):
        """
        Export the model
        """
        return self.model.export_model()

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