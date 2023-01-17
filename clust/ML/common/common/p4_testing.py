import sys
sys.path.append("../")
from Clust.clust.ML.common import p3_training as p3
import pandas as pd 

def getScaledTestData(data, scalerFilePath, scalerParam):
    scaler =None
    result = data
    if scalerParam =='scale':
        if scalerFilePath:
            scaler = getScalerFromFile(scalerFilePath)
            result = getScaledData(data, scaler, scalerParam)
    return result, scaler

def getScalerFromFile(scalerFilePath):
    import joblib
    scaler = joblib.load(scalerFilePath)
    return scaler

def getScaledData(data, scaler, scalerParam):
    if scalerParam=='scale':
        scaledD = pd.DataFrame(scaler.transform(data), index = data.index, columns = data.columns)
    else:
        scaledD = data.copy()
    return scaledD

def getPredictionDFResult(predictions, values, scalerParam, scaler, featureList, target_col):
    print(scalerParam)
    if scalerParam =='scale':
        baseDFforInverse = pd.DataFrame(columns=featureList, index=range(len(predictions)))
        baseDFforInverse[target_col] = predictions
        prediction_inverse = pd.DataFrame(scaler.inverse_transform(baseDFforInverse), columns=featureList, index=baseDFforInverse.index)
        baseDFforInverse[target_col] = values 
        values_inverse = pd.DataFrame(scaler.inverse_transform(baseDFforInverse), columns=featureList, index=baseDFforInverse.index)
        trues = values_inverse[target_col]
        preds = prediction_inverse[target_col]
        df_result = pd.DataFrame(data={"value": trues, "prediction": preds}, index=baseDFforInverse.index)

    else:
        df_result = pd.DataFrame(data={"value": values, "prediction": predictions}, index=range(len(predictions)))
    
    return df_result





def getCleandData(data, cleanTrainDataParam, integration_freq_sec, NaNProcessingParam):
    if cleanTrainDataParam =='Clean':
        import datetime
        timedelta_frequency_sec = datetime.timedelta(seconds= integration_freq_sec)
        result = p3.cleanNaNDF(data, NaNProcessingParam,  timedelta_frequency_sec)

    else:
        result = data.copy()
        pass
    
    return result