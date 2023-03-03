import sys

sys.path.append("..")
sys.path.append("../..")

from Clust.clust.ML.regression_YK.clust_models.rnn_clust import RNNClust
from Clust.clust.ML.regression_YK.clust_models.cnn1d_clust import CNN1DClust
from Clust.clust.ML.regression_YK.clust_models.lstm_fcns_clust import LSTMFCNsClust
from Clust.clust.ML.regression_YK.clust_models.fc_clust import FCClust


class RegressionTest():
    def __init__(self):
        """
        """
        pass

    def set_param(self, params):
        """
        Set Parameter for Test

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

    def set_data(self, test_X, test_y, window_num=0):
        """
        set data for test

        Args:
            test_X (dataframe): Test X data
            test_y (dataframe): Test y data
            window_num (integer) : window size


        Example:

            >>> set_data(test_X, test_y, window_num)
            ...         test_X : test X data
            ...         test_y : test y data
            ...         window_num : window size

        """  
        self.test_loader = self.model.create_testloader(self.batch_size, test_X, test_y, window_num)

    def test(self):
        """
        Test model and return result

        Returns:
            result: model test result
        """
        print("\nStart testing data\n")
        pred, trues, mse, mae = self.model.test(self.params, self.test_loader, self.device)

        return pred, trues, mse, mae
