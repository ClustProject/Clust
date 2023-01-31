import sys
import torch
import numpy as np

sys.path.append("..")
sys.path.append("../..")

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from Clust.clust.transformation.type.DFToNPArray import transDFtoNP, transDFtoNP2
from Clust.clust.ML.common.inference import Inference
from Clust.clust.ML.common.common import p2_dataSelection as p2
from Clust.clust.ML.common.common import p4_testing as p4
from Clust.clust.ML.common import model_manager
from Clust.clust.tool.stats_table import metrics


class RegressionInference(Inference):

    def __init__(self):
        """
        """
        super().__init__()


    def set_param(self, param):
        """

        """
        self.batch_size = param['batch_size']
        self.device = param['device']

    def set_data(self, X, windowNum=0):
        """
        """  
        self.X = transDFtoNP2(X, windowNum)

    def get_result(self, model):
        """

        """

        print("\nStart testing data\n")

        get_loader = self._get_loader()
        preds= self._inference(model, get_loader)
        print(preds)
        print(f'** Dimension of result for test dataset = {preds.shape}')

        return preds





    def _get_loader(self):
        """

        """
        x_data = np.array(self.X)
        test_data = torch.Tensor(x_data)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)

        return test_loader

    def _inference(self, model, test_loader):
        """

        """

        model.eval()   # 모델을 validation mode로 설정
        
        # test_loader에 대하여 검증 진행 (gradient update 방지)
        with torch.no_grad():
            preds = []
            for inputs in test_loader:
                model.to(self.device)
                
                # forward
                # input을 model에 넣어 output을 도출
                outputs = model(inputs)
                
                # 예측 값 및 실제 값 축적
                preds.extend(outputs.detach().cpu().numpy())

        preds = np.array(preds).reshape(-1)

        return preds