import sys
import torch
import torch.nn as nn
import numpy as np
sys.path.append("..")
sys.path.append("../..")

from sklearn.metrics import mean_absolute_error, mean_squared_error 
from Clust.clust.ML.common.inference import Inference
from Clust.clust.transformation.type.DFToNPArray import transDFtoNP2


class ClassificationInference(Inference):
    def __init__(self):
        """
        """
        super().__init__()
        

    def set_param(self, param):
        """

        """
        self.batch_size = param['batch_size']
        self.device = param['device']


    def set_data(self, X, windowNum= 0, dim=None):
        """

        """
        self.X, self.y = transDFtoNP2(X, windowNum, dim)


    
    def get_result(self, model):
        
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
        getTestLoader

        :return: test_loader
        :rtype: DataLoader
        """

        x_data = np.array(self.X)
        test_data = torch.utils.data.TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data))
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=True)

        return test_loader




    def _test(self, model, test_loader):
        """
        Predict classes for test dataset based on the trained model

        :return: predicted classes
        :rtype: numpy array

        :return: prediction probabilities
        :rtype: numpy array

        :return: test accuracy
        :rtype: float
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
        