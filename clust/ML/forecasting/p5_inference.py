import pandas as pd
import sys
sys.path.append("../")

from KETIToolDL.CLUSTTool.common import p4_testing as p4

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
