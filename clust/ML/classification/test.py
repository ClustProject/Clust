import sys
import torch
import torch.nn as nn
import numpy as np
sys.path.append("..")
sys.path.append("../..")

from torch.utils.data import TensorDataset, DataLoader
from Clust.clust.ML.common.inference import Inference
from Clust.clust.transformation.type.DFToNPArray import trans_df_to_np


class ClassificationTest(Inference):
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


    def set_data(self, X, y, windowNum= 0, dim=None):
        """
        set data for test & transform data

        Args:
            test_X (dataframe): Test X data
            test_y (dataframe): Test y data
            window_num (integer) : window size
            dim (integer) : dimension

        Example:

            >>> set_data(test_X, test_y, window_num)
            ...         test_X : test X data
            ...         test_y : test y data
            ...         window_num : window size
            ...         dim : dimension

        """
        self.test_X, self.test_y = trans_df_to_np(X, y, windowNum, dim)


    
    def get_result(self, model):
        """
        Predict RegresiionResult based on model result

        Args:
            model (model) : load train model

        Returns:
            preds (ndarray) : prediction data
            probs (ndarray) : prediction probabilities
            trues (ndarray) : original data
            acc (float) : test accuracy
        
        """
        print("\nStart testing data\n")
        test_loader = self._get_test_loader()
        
        # load best model
        
        # get prediction and accuracy
        preds, probs, trues, acc = self._test(model, test_loader)
        print(f'** Performance of test dataset ==> PROB = {probs}, ACC = {acc}')
        print(f'** Dimension of result for test dataset = {preds.shape}')

        return preds, probs, trues, acc



    def _get_test_loader(self):
        """
        get TestLoader

        Returns:
            test_loader (DataLoader) : data loader
        """

        x_data = np.array(self.test_X)
        y_data = self.test_y
        test_data = TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data))
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)

        return test_loader




    def _test(self, model, test_loader):
        """
        Predict classes for test dataset based on the trained model

        Args:
            model (model) : load train model
            test_loader (DataLoader) : data loader

        Returns:
            preds (ndarray) : prediction data
            probs (ndarray) : prediction probabilities
            trues (ndarray) : original data
            acc (float) : test accuracy
        """
        model.eval()   # 모델을 validation mode로 설정
        
        # test_loader에 대하여 검증 진행 (gradient update 방지)
        with torch.no_grad():
            corrects = 0
            total = 0
            preds = []
            probs = []
            trues = []
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device, dtype=torch.long)

                model.to(self.device)
    
                # forwinputs = inputs.to(device)ard
                # input을 model에 넣어 output을 도출
                outputs = model(inputs)
                prob = outputs
                prob = nn.Softmax(dim=1)(prob)

                # output 중 최댓값의 위치에 해당하는 class로 예측을 수행
                _, pred = torch.max(outputs, 1)
                
                # batch별 정답 개수를 축적함
                corrects += torch.sum(pred == labels.data)
                total += labels.size(0)

                preds.extend(pred.detach().cpu().numpy()) 
                probs.extend(prob.detach().cpu().numpy())
                trues.extend(labels.detach().cpu().numpy())

            preds = np.array(preds)
            probs = np.array(probs)
            trues = np.array(trues)
            
            acc = (corrects.double() / total).item()
        
        return preds, probs, trues, acc
        