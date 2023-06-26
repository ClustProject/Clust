
import sys
import torch
import torch.nn as nn
import numpy as np
sys.path.append("..")
sys.path.append("../..")

from Clust.clust.transformation.type.DFToNPArray import trans_df_to_np_inf

from Clust.clust.ML.classification.classification_model.cnn_1d_model import CNNModel
from Clust.clust.ML.classification.classification_model.fc_model import FCModel
from Clust.clust.ML.classification.classification_model.lstm_fcns_model import LSTMFCNsModel
from Clust.clust.ML.classification.classification_model.rnn_model import RNNModel

class ClassificationInference():
    def __init__(self):
        """
        """
        super().__init__()
        

    def set_param(self, infer_params):
        """
        Set Parameter for Test

        Args:
        param(dict): train parameter


        Example:

            >>> param = { "lr":0.0001,
            ...            "device":"cpu",
            ...            "batch_size":16,
            ...            "n_epochs":10    }
        """
        self.infer_params = infer_params

        
    def set_model(self, model_method, model_file_path, model_params):
        """
        Set model and load weights from model file path

        Args:
            model_method (string): model method name 
            model_file_path (string): path for trained model  
            model_params (dict) : parameter for inference
        """

        self.model_params = model_params
        
        # build initialized model
        if (model_method == 'LSTM_cf') | (model_method == "GRU_cf"):
            self.model = RNNModel(self.model_params)
        elif model_method == 'CNN_1D_cf':
            self.model = CNNModel(self.model_params)
        elif model_method == 'LSTM_FCNs_cf':
            self.model = LSTMFCNsModel(self.model_params)
        elif model_method == 'FC_cf':
            self.model = FCModel(self.model_params)
        else:
            print('Choose the model correctly')

        self.model.load_model(model_file_path)


    def set_data(self, data):
        """
        set data for inference & transform data

        Args:
            data (np.array): Inference data

        """  
        self.inference_loader = self.model.create_inferenceloader(self.infer_params['batch_size'], data)

    def inference(self):
        """
        inference model and return result

        Returns:
            preds (ndarray): prediction data
        """
        print("\nStart inference\n")
        preds = self.model.inference(self.infer_params, self.inference_loader)

        return preds