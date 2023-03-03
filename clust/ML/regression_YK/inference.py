import sys

sys.path.append("..")
sys.path.append("../..")

from Clust.clust.ML.regression_YK.clust_models.rnn_clust import RNNClust
from Clust.clust.ML.regression_YK.clust_models.cnn1d_clust import CNN1DClust
from Clust.clust.ML.regression_YK.clust_models.lstm_fcns_clust import LSTMFCNsClust
from Clust.clust.ML.regression_YK.clust_models.fc_clust import FCClust


class RegressionInference():

    def __init__(self):
        """
        """
        pass

    def set_param(self, params):
        """
        Set Parameter for Inference

        Args:
            param(dict): train parameter


        Example:

            >>> param = { 'num_layers': 2, 
            ...            'hidden_size': 64, 
            ...            'dropout': 0.1,
            ...            'bidirectional': True,
            ...            "lr":0.0001,
            ...            "device":"cpu",
            ...            "batch_size":16,
            ...            "n_epochs":10    }
        """
        self.params = params
        self.batch_size = params['batch_size']
        self.device = params['device']

    def set_model(self, model_method, model_file_path):
        """
        Set model from model file path

        Args:
            model_file_path (string): path for trained model  
        """
        if model_method == 'LSTM_rg':
            self.params['rnn_type'] = 'lstm'
            self.model = RNNClust(self.params)
        elif model_method == 'GRU_rg':
            self.params['rnn_type'] = 'gru'
            self.model = RNNClust(self.params)
        elif model_method == 'CNN_1D_rg':
            self.model = CNN1DClust(self.params)
        elif model_method == 'LSTM_FCNs_rg':
            self.model = LSTMFCNsClust(self.params)
        elif model_method == 'FC_rg':
            self.model = FCClust(self.params)
        else:
            print('Choose the model correctly')

        self.model.load_model(model_file_path)

    def set_data(self, data, window_num=0):
        """
        set data for inference & transform data

        Args:
            data (dataframe): Inference data
            window_num (integer) : window size
    

        Example:

        >>> set_data(test_X, window_num)
        ...         test_X : inference data
        ...         window_num : window size

        """  
        self.inference_loader = self.model.create_inferenceloader(self.batch_size, data, window_num)

    def inference(self):
        """
        """
        print("\nStart inference\n")
        preds = self.model.inference(self.params, self.inference_loader, self.device)

        return preds
