import sys

sys.path.append("..")
sys.path.append("../..")

from Clust.clust.ML.regression.clust_models.rnn_clust import RNNClust
from Clust.clust.ML.regression.clust_models.cnn1d_clust import CNN1DClust
from Clust.clust.ML.regression.clust_models.lstm_fcns_clust import LSTMFCNsClust


class RegressionTest():
    def __init__(self):
        pass

    def set_param(self, test_params):
        """
        Set Parameters for Test

        Args:
            test_params(dict): test parameter

        >>> param = { "device": "cpu",
        ...           "batch_size": 16 }

        """
        self.test_params = test_params

    def set_model(self, model_method, model_file_path, model_params):
        """
        Set model and load weights from model file path

        Args:
            model_method (string): model method name  
            model_file_path (string): path for trained model  
            model_params (dict): hyperparameter for model
        """
        self.model_params = model_params

        if model_method == 'LSTM_rg' or model_method == 'GRU_rg':
            self.model = RNNClust(self.model_params)
        elif model_method == 'CNN_1D_rg':
            self.model = CNN1DClust(self.model_params)
        elif model_method == 'LSTM_FCNs_rg':
            self.model = LSTMFCNsClust(self.model_params)
        else:
            print('Choose the model correctly')

        self.model.load_model(model_file_path)

    def set_data(self, test_X, test_y):
        """
        set data for test

        Args:
            test_X (np.array): Test X data
            test_y (np.array): Test y data


        """  
        self.test_loader = self.model.create_testloader(self.test_params['batch_size'], test_X, test_y)

    def test(self):
        """
        Test model and return result

        Returns:
            nd.array : preds, trues
        """
        print("\nStart testing data\n")
        preds, trues = self.model.test(self.test_params, self.test_loader)

        return preds, trues
