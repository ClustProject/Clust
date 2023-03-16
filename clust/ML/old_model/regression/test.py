import sys
import torch
import numpy as np

sys.path.append("..")
sys.path.append("../..")

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from Clust.clust.transformation.type.DFToNPArray import trans_df_to_np
from Clust.clust.ML.common.inference import Inference
from Clust.clust.ML.common.common import p2_dataSelection as p2
from Clust.clust.ML.common.common import p4_testing as p4
from Clust.clust.ML.common import model_manager
from Clust.clust.tool.stats_table import metrics


class RegressionTest(Inference):

    def __init__(self):
        """
        """
        super().__init__()


    def set_param(self, param):
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
        self.batch_size = param['batch_size']
        self.device = param['device']


    def set_data(self, test_X, test_y, window_num=0):
        """
        set data for test & transform data

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
        self.test_X, self.test_y = trans_df_to_np(test_X, test_y, window_num)


    def get_result(self, model):
        """
        Predict RegresiionResult based on model result

        Args:
            model (model) : load train model

        Returns:
            preds (ndarray) : prediction data
            trues (ndarray) : original data
            mse (float) : mean square error
            mae (float) : mean absolute error
        """

        print("\nStart testing data\n")

        test_loader = self._get_loader()
        pred, trues, mse, mae = self._test(model, test_loader)

        print(f'** Performance of test dataset ==> MSE = {mse}, MAE = {mae}')
        print(f'** Dimension of result for test dataset = {pred.shape}')
        return pred, trues, mse, mae




    def _get_loader(self):
        """

        Returns:
            test_loader (DataLoader) : data loader
        """
        test_data= TensorDataset(torch.Tensor(self.test_X), torch.Tensor(self.test_y))
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)

        return test_loader


    def _test(self, model, test_loader):
        """
        Predict RegresiionResult for test dataset based on the trained model

        Args:
            model (model): load trained model
            test_loader (DataLoader) : data loader

        Returns:
            preds (ndarray) : prediction data
            trues (ndarray) : original data
            mse (float) : mean square error
            mae (float) : mean absolute error
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
