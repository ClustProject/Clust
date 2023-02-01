import sys
import torch
import numpy as np

sys.path.append("..")
sys.path.append("../..")

from torch.utils.data import TensorDataset, DataLoader
from Clust.clust.transformation.type.DFToNPArray import transDFtoNP, transDFtoNP2
from Clust.clust.ML.common.inference import Inference


class RegressionInference(Inference):

    def __init__(self):
        """
        """
        super().__init__()


    def set_param(self, param):
        """
        Set Parameter for Inference

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


    def set_data(self, data, windowNum=0):
        """
        set data for inference & transform data

        Args:
            data (dataframe): Inference data
            window_num (integer) : window size
    

        Example:

        >>> set_data(test_X, window_num)
        ...         test_X : inference data
        ...         window_num : window size

        """  
        self.X = transDFtoNP2(data, windowNum)


    def get_result(self, model):
        """
        Predict RegresiionResult based on model result

        Args:
            model (model) : load trained model

        Returns:
            preds (ndarray) : Inference result data

        """

        print("\nStart testing data\n")

        inference_loader = self._get_loader()
        preds= self._inference(model, inference_loader)
        print(f'** Dimension of result for inference dataset = {preds.shape}')

        return preds



    def _get_loader(self):
        """

        Returns:
            inference_loader (DataLoader) : data loader
        """
        x_data = np.array(self.X)
        inference_data = torch.Tensor(x_data)
        inference_loader = DataLoader(inference_data, batch_size=self.batch_size, shuffle=True)

        return inference_loader

    def _inference(self, model, inference_loader):
        """

        Args:
            model (model): load trained model
            test_loader (DataLoader) : data loader

        Returns:
            preds (ndarray) : Inference result data
        """

        model.eval()   # 모델을 validation mode로 설정
        
        # test_loader에 대하여 검증 진행 (gradient update 방지)
        with torch.no_grad():
            preds = []
            for inputs in inference_loader:
                model.to(self.device)
                
                # forward
                # input을 model에 넣어 output을 도출
                outputs = model(inputs)
                
                # 예측 값 및 실제 값 축적
                preds.extend(outputs.detach().cpu().numpy())

        preds = np.array(preds).reshape(-1)

        return preds