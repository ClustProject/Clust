
import sys
sys.path.append("../")

from Clust.clust.ML.common.common import p2_dataSelection as p2
from Clust.clust.ML.common.common import p4_testing as p4
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

