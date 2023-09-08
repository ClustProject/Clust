import sys

sys.path.append("..")
sys.path.append("../..")

from Clust.clust.ML.anomaly_detection.clust_models.rnn_predictor_clust import RNNAnomalyClust
from Clust.clust.ML.anomaly_detection.clust_models.beatgan_clust import BeatganClust


class AnomalyDetInference():
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

        if model_method == 'LSTM_ad' or model_method == 'GRU_ad':
            self.model = RNNAnomalyClust(self.model_params)
        elif model_method == 'BeatGAN_ad':
            self.model = BeatganClust(self.model_params)
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
