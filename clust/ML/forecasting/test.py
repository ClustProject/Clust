import torch
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"

# from torch.utils.data import DataLoader
from Clust.clust.ML.common.inference import Inference
from Clust.clust.transformation.purpose.machineLearning import  LSTMData
from Clust.clust.ML.forecasting.train import ForecastingTrain as RModel


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