import sys
import torch
import numpy as np

sys.path.append("..")
sys.path.append("../..")

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from Clust.clust.transformation.type.DFToNPArray import transDFtoNP
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


    def set_data(self, X, y, windowNum=0):
        """
        """  
        self.X, self.y = transDFtoNP(X, y, windowNum)


    def get_result(self, model):
        """

        """

        print("\nStart testing data\n")

        test_loader = self._get_test_loader()
        pred, trues, mse, mae = self._test(model, test_loader)

        print(f'** Performance of test dataset ==> MSE = {mse}, MAE = {mae}')
        print(f'** Dimension of result for test dataset = {pred.shape}')
        return pred, trues



    def _get_test_loader(self):
        """

        """
        x_data = np.array(self.X)
        y_data = self.y
        test_data= TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data))
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
        return test_loader


    def _test(self, model, test_loader):
        """

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




# ------------------------ 질문 -----------------------
"""
- example에서 test와 inference의 차이가 들어오는 y의 길이던데 맞나요?
- test의 경우는 기존에 나눈 testy길이, inference의 경우는 1개만 들어오고 있습니다
- 그리고 regression의 경우에는 y값이 없는게 아니라 독립변수가 있어야하는게 아닌가요?

"""







def get_test_result(dataName_X, dataName_y, DataMeta, ModelMeta, dataFolderPath, mode_select='test', window_num=0, db_client=None):

    dataSaveMode_X = DataMeta[dataName_X]["integrationInfo"]["DataSaveMode"]
    dataSaveMode_y = DataMeta[dataName_y]["integrationInfo"]["DataSaveMode"]
    dataX = p2.getSavedIntegratedData(dataSaveMode_X, dataName_X, dataFolderPath)
    datay = p2.getSavedIntegratedData(dataSaveMode_y, dataName_y, dataFolderPath)

    X_scalerFilePath = ModelMeta['files']['XScalerFile']["filePath"]
    y_scalerFilePath = ModelMeta['files']['yScalerFile']["filePath"]
    modelFilePath = ModelMeta['files']['modelFile']["filePath"]

    featureList = ModelMeta["featureList"]
    target = ModelMeta["target"]
    scalerParam = ModelMeta["scalerParam"]
    model_method = ModelMeta["model_method"]
    # ModelMeta["trainParameter"]['batch_size'] = 1
    trainParameter = ModelMeta["trainParameter"]


    # Scaling Test Input
    if mode_select == 'test':
        dataX = p2.getSavedIntegratedData(dataSaveMode_X, dataName_X, dataFolderPath)
        datay = p2.getSavedIntegratedData(dataSaveMode_y, dataName_y, dataFolderPath)
    elif mode_select == 'inference':
        dataX = p2.getSavedIntegratedData(dataSaveMode_X, dataName_X, dataFolderPath)[:window_num]
        datay = p2.getSavedIntegratedData(dataSaveMode_y, dataName_y, dataFolderPath)[:1]

        
    test_x, scaler_X = p4.getScaledTestData(dataX[featureList], X_scalerFilePath, scalerParam)
    test_y, scaler_y = p4.getScaledTestData(datay[target], y_scalerFilePath, scalerParam)
    
    # 4. Testing
    # batch_size=1
    df_result, result_metrics = get_result_metrics(trainParameter, test_x, test_y, target, modelFilePath, scalerParam, scaler_y, windowNum=0)
    return df_result, result_metrics


# Test
def get_result_metrics(trainParameter, test_x, test_y, target, model_file_path, scalerParam, scaler_y, windowNum=0):

    ri = RegressionInference()
    ri.set_param(trainParameter)
    ri.set_data(test_x, test_y, windowNum)
    model = model_manager.load_pickle_model(model_file_path)
    pred, trues =  ri.get_result(model)

    df_result = p4.getPredictionDFResult(pred, trues, scalerParam, scaler_y, featureList= target, target_col = target[0])
    # df_result.index = test_y.index

    result_metrics =  metrics.calculate_metrics_df(df_result)

    return df_result, result_metrics







def regression_inference():



    return



def regression_test_result():




    return 