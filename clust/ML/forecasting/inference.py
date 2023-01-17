import torch
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"

# from torch.utils.data import DataLoader
from Clust.clust.ML.common.inference import Inference
from Clust.clust.transformation.purpose.machineLearning import  LSTMData
from Clust.clust.ML.forecasting.train import RNNStyleModelTrainer as RModel

class RNNStyleModelInfernce(Inference):
    # For small data without answer
    def __init__(self):
        """
            example: 
            >>> from KETIToolDL.PredictionTool.RNNStyleModel.inference import RNNStyleModelInfernce
            >>> Inference = RNNStyleModelInfernce()
            >>> inference_input_scaledTensor = Inference.getTensorInput(inference_input_scaled)
            >>> Inference.setData(inference_input_scaledTensor)
            >>> Inference.setModel(trainParameter, model_method, modelFilePath)
            >>> inference_result = Inference.get_result()
            >>> print(inference_result)
            
        """
        self.batch_size = 1
    
    def set_param(self):
        pass

    def set_data(self):
        pass

    def get_result(self):
        pass

    def test(self):
        pass

    def setModel(self, trainParameter, model_method, modelFilePath):
        IM= RModel()
        IM.setTrainParameter(trainParameter)
        IM.getModel(model_method)
        self.infModel = IM.model
        self.infModel.load_state_dict(torch.load(modelFilePath[0])) # gpu환경에서 훈련한 모델을 cpu 환경에서 실행시, map_location='cpu' 필요
        self.infModel.eval()

    def getTensorInput(self, data):
        """The method get tensor input for training.

        dataframe -> ndarray size = (past_step, len(feature_col_list)) -> 
        ndarray size = (1, past_step, len(feature_col_list))  -> torch tensor

        Note:
            tensor input transformation

        Args:
            data (pd.DataFrame): input dataframe

        Returns:
            data (torch.utils.data.DataLoader): input tensor data
        """
        inference_input = data.values.astype(np.float32)
        inference_input = inference_input.reshape((-1, inference_input.shape[0], inference_input.shape[1]))
        inference_input_tensor = torch.tensor(inference_input)
        return inference_input_tensor

    def setData(self, data):
        self.data = data.to(device)
  
    def get_result(self):
        yhat = self.infModel(self.data)
        result = yhat.to('cpu').detach().numpy()
        return result



class RNNStyleModelTestInference(Inference):
    # For TestData with answer
    def __init__(self):
        self.batch_size = 1
    
    def setModel(self, trainParameter, model_method, modelFilePath):
        IM= RModel()
        IM.setTrainParameter(trainParameter)
        IM.getModel(model_method)
        self.infModel = IM.model
        self.infModel.load_state_dict(torch.load(modelFilePath[0]))

    def setTestData(self, test, transformParameter, cleanParam):
        self.input_dim = len(transformParameter['feature_col'])
        LSTMD = LSTMData()
        testX_arr, testy_arr = LSTMD.transformXyArr(test, transformParameter, cleanParam)
        self.test_DataSet, self.test_loader = LSTMD.getTorchLoader(testX_arr, testy_arr, self.batch_size)
        #test_loader_one = DataLoader(test_DataSet, batch_size=1, shuffle=False, drop_last=True)


    def get_result(self):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in self.test_loader:
                x_test = x_test.view([self.batch_size, -1, self.input_dim]).to(device)
                y_test = y_test.to(device)
                self.infModel.eval()
                yhat = self.infModel(x_test)
                predictions.append(yhat.to('cpu').detach().numpy().ravel()[0])
                values.append(y_test.to('cpu').detach().numpy().ravel()[0])
        return predictions, values

