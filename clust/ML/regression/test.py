import sys
sys.path.append("../")
sys.path.append("../../")

from Clust.clust.ML.common.common import p2_dataSelection as p2
from Clust.clust.ML.common.common import p4_testing as p4
from Clust.clust.ML.regression.train import RegressionML as RML
from Clust.clust.ML.regression.inference import RegressionModelTestInference as RTI
from Clust.clust.tool.stats_table import metrics


def get_test_result(dataName_X, dataName_y, DataMeta, ModelMeta, dataFolderPath, windowNum=0, db_client=None):

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
    ModelMeta["trainParameter"]['batch_size'] = 1
    trainParameter = ModelMeta["trainParameter"]


    # Scaling Test Input
    test_x, scaler_X = p4.getScaledTestData(dataX[featureList], X_scalerFilePath, scalerParam)
    test_y, scaler_y = p4.getScaledTestData(datay[target], y_scalerFilePath, scalerParam)
    
    # 4. Testing
    # batch_size=1
    df_result, result_metrics = get_result_metrics(trainParameter, test_x, test_y, model_method, target, modelFilePath, scalerParam, scaler_y, windowNum=0)
    return df_result, result_metrics





def get_result_metrics(trainParameter, test_x, test_y, model_method, target, modelFilePath, scalerParam, scaler_y, windowNum=0):

    rml = RML()
    rml.set_param(trainParameter)
    model = rml.get_model(model_method)

    ri = RTI()
    ri.set_param(trainParameter)
    ri.set_data(test_x, test_y, windowNum)
    pred, trues, mse, mae =  ri.get_result(model, modelFilePath)

    df_result = p4.getPredictionDFResult(pred, trues, scalerParam, scaler_y, featureList= target, target_col = target[0])
    df_result.index = test_y.index

    result_metrics =  metrics.calculate_metrics_df(df_result)

    return df_result, result_metrics





















"""
def getTestResult(dataName_X, dataName_y, modelName, DataMeta, ModelMeta, dataFolderPath, currentFolderPath, device, windowNum=0, db_client=None):

    dataSaveMode_X = DataMeta[dataName_X]["integrationInfo"]["DataSaveMode"]
    dataSaveMode_y = DataMeta[dataName_y]["integrationInfo"]["DataSaveMode"]
    dataX = p2.getSavedIntegratedData(dataSaveMode_X, dataName_X, dataFolderPath)
    datay = p2.getSavedIntegratedData(dataSaveMode_y, dataName_y, dataFolderPath)
    X_scalerFilePath = os.path.join(currentFolderPath, ModelMeta[modelName]['files']['XScalerFile']["filePath"])
    y_scalerFilePath = os.path.join(currentFolderPath, ModelMeta[modelName]['files']['yScalerFile']["filePath"])
    modelFilePath_old = ModelMeta[modelName]['files']['modelFile']["filePath"]
    modelFilePath =[]
    for modelFilePath_one in modelFilePath_old:
        modelFilePath.append(os.path.join(currentFolderPath, modelFilePath_one))
    featureList = ModelMeta[modelName]["featureList"]
    target = ModelMeta[modelName]["target"]
    scalerParam = ModelMeta[modelName]["scalerParam"]
    model_method = ModelMeta[modelName]["model_method"]
    trainParameter = ModelMeta[modelName]["trainParameter"]
    

    # Scaling Test Input
    test_x, scaler_X = p4.getScaledTestData(dataX[featureList], X_scalerFilePath, scalerParam)
    test_y, scaler_y = p4.getScaledTestData(datay[target], y_scalerFilePath, scalerParam)
    # 4. Testing
    batch_size=1
    df_result, result_metrics = getResultMetrics(test_x, test_y, model_method, target, modelFilePath, scalerParam, scaler_y, trainParameter, batch_size, device, windowNum)
    return df_result, result_metrics

"""