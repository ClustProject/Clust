import torch
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"

from Clust.clust.ML.common.inference import Inference
from torch.utils.data import DataLoader

class ForecastingInfernce(Inference):
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
        set data for inference & transform data

        Args:
            data (dataframe): Inference data
    

        Example:

        >>> set_data(data)
        ...         data : inference data

        """
        data = data.values.astype(np.float32)
        self.data = data.reshape((-1, data.shape[0], data.shape[1]))


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
        preds = self._inference(model, inference_loader)

        return preds



    def _get_loader(self):
        """

        Returns:
            inference_loader (DataLoader) : data loader
        """
        data = torch.Tensor(self.data)
        inference_loader = DataLoader(data, batch_size=self.batch_size, shuffle=False, drop_last=True)

        return inference_loader
        

    def _inference(self, model, inference_loader):
        """


        Args:
            model (model): load trained model
            test_loader (DataLoader) : data loader

        Returns:
            preds (ndarray) : Inference result data
        """

        model.eval()

        with torch.no_grad():
            preds = []

            for input in inference_loader:

                inputs = input.to(device)

                model.to(device)

                outputs = model(inputs)

                preds.extend(outputs.detach().numpy())

        preds = np.array(preds).reshape(-1)

        return preds


