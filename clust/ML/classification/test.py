import sys
sys.path.append("..")
sys.path.append("../..")

from Clust.clust.ML.classification.classification_model.cnn_1d_model import CNNModel
from Clust.clust.ML.classification.classification_model.fc_model import FCModel
from Clust.clust.ML.classification.classification_model.lstm_fcns_model import LSTMFCNsModel
from Clust.clust.ML.classification.classification_model.rnn_model import RNNModel


class ClassificationTest():
    def __init__(self):
        """
        """
        super().__init__()
        

    def set_param(self, test_params):
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
        self.test_params = test_params


    def set_model(self, model_method, model_file_path, model_params):
        """
        Build model and return initialized model for selected model_name

        Args:
            model_method (string): model method name
            model_file_path (string): path for trained model  
            model_params (dict) : parameter for test
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
            preds (ndarray): prediction data
            trues (ndarray): original data
            mse (float): mean square error  # TBD
            mae (float): mean absolute error    # TBD
        """
        print("\nStart testing data\n")
        preds, probs, trues, acc = self.model.test(self.test_params, self.test_loader)

        return preds, probs, trues, acc

        