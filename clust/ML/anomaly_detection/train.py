import torch
import numpy as np
import random
from pathlib import Path

from Clust.clust.ML.anomaly_detection.clust_models.rnn_predictor_clust import RNNAnomalyClust
from Clust.clust.ML.anomaly_detection.clust_models.beatgan_clust import BeatganClust
from Clust.clust.ML.anomaly_detection.clust_models.AT_clust import ATClust
from Clust.clust.ML.anomaly_detection.clust_models.lstm_clust import LSTMClust

# Set the random seed manually for reproducibility.
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

class AnomalyDetTrain():
    def __init__(self):
        pass

    def set_param(self, train_params):
        """
        Set parameters for train

        Args:
            params (dict): parameters for train
        """
        self.train_params = train_params

    def set_model(self, model_method, model_params, model_file_path=None):
        """
        Set model for selected model_method

        Args:
            model_method (string): model method name  
            model_params (dict) : hyperparameter for model
        """
        self.model_params = model_params

        if model_method == 'LSTM_ad' or model_method == 'GRU_ad':
            self.model = RNNAnomalyClust(self.model_params)
        elif model_method == 'BeatGAN_ad':
            self.model = BeatganClust(self.model_params)
        elif model_method == 'AT_ad':
            self.model = ATClust(self.model_params)
        elif model_method == 'LSTM_VAE_ad':
            self.model = LSTMClust(self.model_params)
        else:
            print('Choose the model correctly')

        # if self.train_params['resume'] or self.train_params['pretrained']:
        #     # model_file_path 는 AD_pipeline 에서?
        #     self.model.load_model(model_file_path)

    def set_data(self, train_X, train_y, val_X, val_y):
        """
        set train, val data & transform data for training

        Args:
            train_x (np.array): train X data
            train_y (np.array): train y data (optional)
            val_x (np.array): validation X data
            val_y (np.array): validation y data (optional)
        """
        self.train_loader, self.valid_loader = self.model.create_trainloader(self.train_params['batch_size'], train_X, train_y, val_X, val_y)

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
