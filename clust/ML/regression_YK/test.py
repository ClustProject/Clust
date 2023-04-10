import sys

sys.path.append("..")
sys.path.append("../..")

from Clust.clust.ML.regression_YK.clust_models.rnn_clust import RNNClust
from Clust.clust.ML.regression_YK.clust_models.cnn1d_clust import CNN1DClust
from Clust.clust.ML.regression_YK.clust_models.lstm_fcns_clust import LSTMFCNsClust
from Clust.clust.ML.regression_YK.clust_models.fc_clust import FCClust


class RegressionTest():
    def __init__(self):
        pass

    def set_param(self, test_params):
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
        # TODO: parameter refactoring
        self.test_params = test_params
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

        self.model.load_model(model_file_path)

    def set_data(self, test_X, test_y):
        """
        set data for test

        Args:
            test_X (dataframe): Test X data
            test_y (dataframe): Test y data


        Example:

            >>> set_data(test_X, test_y, window_num)
            ...         test_X : test X data
            ...         test_y : test y data

        """  
        self.test_loader = self.model.create_testloader(self.test_params['batch_size'], test_X, test_y)

    def test(self):
        """
        Test model and return result

        Returns:
            preds (ndarray): prediction data
            trues (ndarray): original data
        """
        print("\nStart testing data\n")
        preds, trues = self.model.test(self.test_params, self.test_loader)

        return preds, trues
