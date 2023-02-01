import pandas as pd
import sys
sys.path.append("../")

from KETIToolDL.CLUSTTool.common import p4_testing as p4
from Clust.clust.tool.stats_table import metrics

def getTestValues(test, trainParameter, transformParameter, model_method, modelFilePath, cleanParam):
    from KETIToolDL.PredictionTool.RNNStyleModel.inference import RNNStyleModelTestInference
    TestInference = RNNStyleModelTestInference()
    TestInference.setTestData(test, transformParameter, cleanParam)
    TestInference.setModel(trainParameter, model_method, modelFilePath)
    predictions, values = TestInference.get_result()

    return predictions, values

def getTestResult(dataName, modelName, DataMeta, ModelMeta, dataRoot, db_client):

    dataSaveMode = DataMeta[dataName]["integrationInfo"]["DataSaveMode"]
    data = p2.getSavedIntegratedData(dataSaveMode, dataName, dataRoot, db_client)

    scalerFilePath = ModelMeta[modelName]['files']['scalerFile']["filePath"]
    modelFilePath = ModelMeta[modelName]['files']['modelFile']["filePath"]
    featureList = ModelMeta[modelName]["featureList"]
    cleanTrainDataParam = ModelMeta[modelName]["cleanTrainDataParam"]
    scalerParam = ModelMeta[modelName]["scalerParam"]
    integration_freq_sec = ModelMeta[modelName]['trainDataInfo']["integration_freq_sec"]
    NaNProcessingParam = ModelMeta[modelName]['NaNProcessingParam']
    trainParameter = ModelMeta[modelName]["trainParameter"]
    transformParameter = ModelMeta[modelName]["transformParameter"]
    model_method = ModelMeta[modelName]["model_method"]
    target_col = ModelMeta[modelName]["transformParameter"]["target_col"]

    test, scaler = p4.getScaledTestData(data[featureList], scalerFilePath, scalerParam)
    
    test = p4.getCleandData(test, cleanTrainDataParam, integration_freq_sec, NaNProcessingParam)
    
    prediction, values = getTestValues(test, trainParameter, transformParameter, model_method, modelFilePath, cleanTrainDataParam)
    df_result = p4.getPredictionDFResult(prediction, values, scalerParam, scaler, featureList, target_col)
    df_result.index = test[(transformParameter['future_step']+transformParameter['past_step']-1):].index

    """
    df_result = p4.refineData(df_result)
    
    df_result.index = test.index
    """
    result_metrics =  metrics.calculate_metrics_df(df_result)

    return df_result, result_metrics





def inference(input, trainParameter, model_method, modelFilePath, scalerParam, scalerFilePath, featureList, target_col):
    inputDF = pd.DataFrame(input, columns = featureList)
    # 4.Inference Data Preparation
    inputData, scaler = p4.getScaledTestData(inputDF[featureList], scalerFilePath, scalerParam)

    # 5. Inference
    from KETIToolDL.PredictionTool.RNNStyleModel.inference import RNNStyleModelInfernce
    Inference = RNNStyleModelInfernce()
    input_DTensor = Inference.getTensorInput(inputData)
    Inference.setData(input_DTensor)

    Inference.setModel(trainParameter, model_method, modelFilePath)
    inference_result = Inference.get_result()
    print(inference_result)
    
    if scalerParam =='scale':
        baseDFforInverse = pd.DataFrame(columns=featureList, index=range(1))
        baseDFforInverse[target_col] = inference_result[0]
        prediction_inverse = pd.DataFrame(scaler.inverse_transform(baseDFforInverse), columns=featureList, index=baseDFforInverse.index)
        result = prediction_inverse[target_col].values[0]
    else:
        result = inference_result[0][0]

    return result

