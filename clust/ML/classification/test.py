import os
import sys
import pandas as pd

sys.path.append("../")

from sklearn.metrics import classification_report
from Clust.clust.ML.common.common import p2_dataSelection as p2
from Clust.clust.ML.common.common import p4_testing as p4
from Clust.clust.ML.classification.train import ClassificationML as CML
from Clust.clust.ML.classification.inference import ClassificationModelTestInference as CTI

def getTestResult(dataName_X, dataName_y, modelName, DataMeta, ModelMeta, dataFolderPath, device, windowNum=0, db_client=None):
    dataSaveMode_X = DataMeta[dataName_X]["integrationInfo"]["DataSaveMode"]
    dataSaveMode_y = DataMeta[dataName_y]["integrationInfo"]["DataSaveMode"]
    dataX = p2.getSavedIntegratedData(dataSaveMode_X, dataName_X, dataFolderPath)
    datay = p2.getSavedIntegratedData(dataSaveMode_y, dataName_y, dataFolderPath)
    
    X_scalerFilePath = ModelMeta[modelName]['files']['XScalerFile']["filePath"]
    y_scalerFilePath = ModelMeta[modelName]['files']['yScalerFile']["filePath"]
    modelFilePath = ModelMeta[modelName]['files']['modelFile']["filePath"]

    featureList = ModelMeta[modelName]["featureList"]
    target = ModelMeta[modelName]["target"]
    scalerParam = ModelMeta[modelName]["scalerParam"]
    model_method = ModelMeta[modelName]["model_method"]
    trainParameter = ModelMeta[modelName]["trainParameter"]
    
    dim = None
    if model_method == "FC_cf":
        dim = 2
    
    # Scaling Test Input

    test_x, scaler_X = p4.getScaledTestData(dataX[featureList], X_scalerFilePath, scalerParam)
    test_y, scaler_y = p4.getScaledTestData(datay[target], y_scalerFilePath, scalerParam)# No Scale
    # test_y = datay[target] # for classification

    # 4. Testing
    batch_size=1
    scalerParam="noScale" # for classification
    df_result, result_metrics, acc = getResultMetrics(test_x, test_y, model_method, target, modelFilePath, scalerParam, scaler_y, trainParameter, batch_size, device, windowNum, dim)
    return df_result, result_metrics, acc


def getResultMetrics(test_x, test_y, model_method, target, modelFilePath, scalerParam, scaler_y, trainParameter, batch_size, device, windowNum=0, dim= None):
    cml = CML(model_method, trainParameter)
    model = cml.getModel()

    ci = CTI(test_x, test_y, batch_size, device)
    ci.transInputDFtoNP(windowNum, dim)
    pred, prob, trues, acc = ci.get_result(model, modelFilePath)
    result_metrics = classification_report(trues, pred, output_dict = True)
    
    df_result = p4.getPredictionDFResult(pred, trues, scalerParam, scaler_y, featureList= target, target_col = target[0])
    df_result.index = test_y.index
    
    return df_result, result_metrics, acc

 