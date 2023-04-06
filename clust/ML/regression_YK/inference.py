import sys

sys.path.append("..")
sys.path.append("../..")

from Clust.clust.ML.regression_YK.clust_models.rnn_clust import RNNClust
from Clust.clust.ML.regression_YK.clust_models.cnn1d_clust import CNN1DClust
from Clust.clust.ML.regression_YK.clust_models.lstm_fcns_clust import LSTMFCNsClust
from Clust.clust.ML.regression_YK.clust_models.fc_clust import FCClust


class RegressionInference():
    def __init__(self):
        pass

    def set_param(self, infer_params):
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
        self.infer_params = infer_params
        # self.batch_size = params['batch_size']
        # self.device = params['device']

    def set_model(self, model_method, model_file_path, model_params):
        """
        Set model and load weights from model file path

        Args:
            model_method (string): model method name  
            model_file_path (string): path for trained model  
            model_params (dict): parameters to create a model
        """
        self.model_params = model_params

        if model_method == 'LSTM' or 'GRU' or 'RNN':
            self.model = RNNClust(self.model_params)
        elif model_method == 'CNN_1D':
            self.model = CNN1DClust(self.model_params)
        elif model_method == 'LSTM_FCNs':
            self.model = LSTMFCNsClust(self.model_params)
        elif model_method == 'FC':
            self.model = FCClust(self.model_params)
        else:
            print('Choose the model correctly')

        self.model.load_model(model_file_path)

    def set_data(self, data):
        """
        set data for inference & transform data

        Args:
            data (dataframe): Inference data
    
        Example:

        >>> set_data(test_X, window_num)
        ...         test_X : inference data

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
