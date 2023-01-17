import sys
import torch
import numpy as np
import torch.nn as nn

sys.path.append("..")
sys.path.append("../..")

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from Clust.clust.ML.common.inference import Inference
from Clust.clust.transformation.type.DFToNPArray import transDFtoNP
from Clust.clust.ML.common import model_manager


class RegressionModelTestInference(Inference):

    def __init__(self):
        """
        Set initial parameter for inference

        :param batch_size: BatchSize for inference
        :type batch_size: Integer
        
        :param device: device specification for inference
        :type device: String
        """
        super().__init__()

    def set_param(self, param):
        """

        """
        self.param = param
        self.batch_size = param['batch_size']
        self.device = param['device']

    def set_data(self, X, y, windowNum=0):
        """
        """
        # windowNum, dim = self._check_win_dim()
        self.X, self.y = transDFtoNP(X, y, windowNum)


    def get_result(self, model, model_path):
        """
        Predict RegresiionResult based on model result
        :param init_model: initialized model
        :type model: model

        :param best_model_path: path for loading the best trained model
        :type best_model_path: str

        :return: predicted values
        :rtype: numpy array

        :return: test mse
        :rtype: float

        :return: test mae
        :rtype: float
        """

        print("\nStart testing data\n")
        test_loader = self._get_test_loader()
        # load best model
        load_model = model_manager.load_pickle_model(model_path[0])
        model.load_state_dict(load_model)

        # get prediction and accuracy
        pred, trues, mse, mae = self._test(model, test_loader)
        print(f'** Performance of test dataset ==> MSE = {mse}, MAE = {mae}')
        print(f'** Dimension of result for test dataset = {pred.shape}')
        return pred, trues, mse, mae



    def _check_win_dim(self):
        if 'windowNum' in self.param.keys():
            windowNum = self.param['windowNum']
        else:
            windowNum = 0

        if 'dim' in self.param.keys():
            dim = self.param['dim']
        else:
            dim = None
        return windowNum, dim



    def _get_test_loader(self):
        """
        getTestLoader

        :return: test_loader
        :rtype: DataLoader
        """
        x_data = np.array(self.X)
        y_data = self.y
        testData= TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data))
        test_loader = DataLoader(testData, batch_size=self.batch_size, shuffle=True)
        return test_loader



    def _test(self, model, test_loader):
        """
        Predict RegresiionResult for test dataset based on the trained model

        :param model: best trained model
        :type model: model

        :param test_loader: test dataloader
        :type test_loader: DataLoader

        :return: predicted values
        :rtype: numpy array

        :return: test mse
        :rtype: float

        :return: test mae
        :rtype: float
        """

        model.eval()   # 모델을 validation mode로 설정
        
        # test_loader에 대하여 검증 진행 (gradient update 방지)
        with torch.no_grad():
            trues, preds = [], []

            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device, dtype=torch.float)

                model.to(self.device)
                
                # forward
                # input을 model에 넣어 output을 도출
                outputs = model(inputs)
                
                # 예측 값 및 실제 값 축적
                trues.extend(labels.detach().cpu().numpy())
                preds.extend(outputs.detach().cpu().numpy())
        
        preds = np.array(preds).reshape(-1)
        trues = np.array(trues)

        mse = mean_squared_error(trues, preds)
        mae = mean_absolute_error(trues, preds)
        return preds, trues, mse, mae

    