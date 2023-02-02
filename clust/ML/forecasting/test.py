import torch
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"

from torch.utils.data import TensorDataset, DataLoader
from Clust.clust.ML.common.inference import Inference
from Clust.clust.transformation.purpose.machineLearning import LSTMData


class ForecasatingTest(Inference):
    def __init__(self):
        """
        
        """
        super().__init__()


    def set_param(self, param):
        """
        Set Parameter for Inference

        use all model meta

        Args:
            param(dict): model meta


        Example:

            >>> param = { 'trainDataInfo': {...}, 
            ...            'cleanTrainDataParam': {...}, 
            ...            'transformParameter': {...},
            ...            ...
            ...            ...                 }

        """

        self.param = param
        self.clean_param = param['cleanTrainDataParam']
        self.transform_parameter = param['transformParameter']
        self.input_dim = len(self.transform_parameter['feature_col'])
        self.batch_size = 1


    def set_data(self, data):
        """
        set data for test & transform data

        Args:
            data (dataframe): Inference data
    

        Example:

        >>> set_data(data)
        ...         data : inference data

        """
        LSTMD = LSTMData()
        self.testX_arr, self.testy_arr = LSTMD.transformXyArr(data, self.transform_parameter, self.clean_param )




    def get_result(self, model):
        """
        Predict RegresiionResult based on model result

        Args:
            model (model) : load trained model

        Returns:
            preds (ndarray) : prediction data
            trues (ndarray) : original data
        
        """
        print("\nStart testing data\n")

        test_loader = self._get_loader()
        preds, trues = self._test(model, test_loader)

        return preds, trues



    def _get_loader(self):
        """

        Returns:
            test_loader (DataLoader) : data loader
        """
        features = torch.Tensor(self.testX_arr)
        targets = torch.Tensor(self.testy_arr)

        test_dataSet = TensorDataset(features, targets)
        test_loader = DataLoader(test_dataSet, batch_size=self.batch_size, shuffle=False, drop_last=True)
        print("features shape:", features.shape, "targets shape: ", targets.shape)

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
        """

        model.eval()

        with torch.no_grad():
            preds, trues = [], []

            for x_test, y_test in test_loader:

                x_test = x_test.view([self.batch_size, -1, self.input_dim]).to(device)
                y_test = y_test.to(device)

                model.to(device)

                outputs = model(x_test)

                preds.extend(outputs.detach().numpy())
                trues.extend(y_test.detach().numpy())

        preds = np.array(preds).reshape(-1)
        trues = np.array(trues).reshape(-1)
        
        return preds, trues


                # preds.append(outputs.detach().numpy().ravel()[0])
                # trues.append(y_test.detach().numpy().ravel()[0])