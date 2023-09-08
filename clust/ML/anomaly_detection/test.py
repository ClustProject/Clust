import sys

sys.path.append("..")
sys.path.append("../..")

from Clust.clust.ML.anomaly_detection.clust_models.rnn_predictor_clust import RNNAnomalyClust
from Clust.clust.ML.anomaly_detection.clust_models.beatgan_clust import BeatganClust


class AnomalyDetTest():
    def __init__(self):
        pass

    def set_param(self, test_params):
        """
        Set Parameters for Test

        Args:
        test_params(dict): test parameter

        Example:

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

        if model_method == 'LSTM_ad' or model_method == 'GRU_ad':
            self.model = RNNAnomalyClust(self.model_params)
        elif model_method == 'BeatGAN_ad':
            self.model = BeatganClust(self.model_params)
        else:
            print('Choose the model correctly')

        self.model.load_model(model_file_path)

    def set_data(self, test_X, test_y):
        """
        set data for test

        Args:
            test_X (np.array): Test X data
            test_y (np.array): Test y data ## y를 reconstruction 이면 복원할 데이터, forecasting 이면 t+1 시점 적용
            test_label : 이상치 탐지를 위한 실제 이상치 여부를 확인하는 label


        """  
        self.test_loader = self.model.create_testloader(self.test_params['batch_size'], test_X, test_y)

    def test(self):
        """
        Test model and return result
        # 각모델마다 결과값이 다를 것 같은데, 결과 통일 했던 f1_score 추가해주세요.
        Returns:
            preds (ndarray): prediction data
            trues (ndarray): original data
            
        """
        print("\nStart testing data\n")
        preds, trues = self.model.test(self.test_params, self.test_loader)

        return preds, trues
