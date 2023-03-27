import sys

sys.path.append("..")
sys.path.append("../..")

from Clust.clust.ML.regression_YK.clust_models.rnn_forecast import RNNForecast


class ForecastTest():
    def __init__(self):
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
        self.clean_param = params['cleanTrainDataParam']
        self.transform_parameter = params['transformParameter']
        self.input_dim = len(self.transform_parameter['feature_col'])
        # need batch size?
        self.params['batch_size'] = 1
        self.batch_size = self.params['batch_size']
        # device parameter? 
        self.params['device'] = 'cpu'
        self.device = self.params['device']

    def set_model(self, model_method, model_file_path):
        """
        Set model and load weights from model file path

        Args:
            model_method (string): model method name  
            model_file_path (string): path for trained model  
        """
        if model_method == 'rnn':
            self.params['rnn_type'] = 'rnn'
            self.model = RNNForecast(self.params)
        elif model_method == 'lstm':
            self.params['rnn_type'] = 'lstm'
            self.model = RNNForecast(self.params)
        elif model_method == 'gru':
            self.params['rnn_type'] = 'gru'
            self.model = RNNForecast(self.params)
        else:
            print('Choose the model correctly')

        self.model.load_model(model_file_path)

    def set_data(self, test_X):
        """
        set data for test

        Args:
            test_X (dataframe): Test X data


        Example:

            >>> set_data(test_X)
            ...         test_X : test X data

        """  
        self.test_loader = self.model.create_testloader(self.batch_size, test_X)

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

        pred, trues = self.model.test(self.params, self.test_loader, self.device)

        return pred, trues
