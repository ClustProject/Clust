import sys

sys.path.append("..")
sys.path.append("../..")

from Clust.clust.ML.regression.clust_models.rnn_clust import RNNClust
from Clust.clust.ML.regression.clust_models.cnn1d_clust import CNN1DClust
from Clust.clust.ML.regression.clust_models.lstm_fcns_clust import LSTMFCNsClust
from Clust.clust.ML.regression.clust_models.fc_clust import FCClust


class RegressionInference():
    def __init__(self):
        pass

    def set_param(self, infer_params):
        """
        Set Parameters for Inference

        Args:
            infer_params(dict): inference parameter

        Example:

            >>> param = { "device": "cpu",
            ...           "batch_size": 1 }
        """
        self.infer_params = infer_params

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
        elif model_method == 'FC_rg':
            self.model = FCClust(self.model_params)
        else:
            print('Choose the model correctly')

        self.model.load_model(model_file_path)

    def set_data(self, infer_X):
        """
        set data for inference & transform data

        Args:
            infer_X (np.array): Inference data
    
        """  
        self.inference_loader = self.model.create_inferenceloader(self.infer_params['batch_size'], infer_X)

    def inference(self):
        """
        inference model and return result

        Returns:
            preds (ndarray): prediction data
        """
        print("\nStart inference\n") 
        preds = self.model.inference(self.infer_params, self.inference_loader)

        return preds
