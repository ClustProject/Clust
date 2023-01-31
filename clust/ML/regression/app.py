import sys
import pandas as pd 
sys.path.append("../")
sys.path.append("../../")

from Clust.clust.ML.common import model_manager
from Clust.clust.ML.common.common import p2_dataSelection as p2
from Clust.clust.ML.common.common import p4_testing as p4
from Clust.clust.ML.regression.test import RegressionTest as RT
from Clust.clust.ML.regression.inference import RegressionInference as RI
from Clust.clust.tool.stats_table import metrics

# Test
def get_test_result(data_name_X, data_name_y, data_meta, model_meta, data_folder_path=None, window_num=0, db_client=None):

    data_save_mode_X = data_meta[data_name_X]["integrationInfo"]["DataSaveMode"]
    data_save_mode_y = data_meta[data_name_y]["integrationInfo"]["DataSaveMode"]
    data_X = p2.getSavedIntegratedData(data_save_mode_X, data_name_X, data_folder_path)
    data_y = p2.getSavedIntegratedData(data_save_mode_y, data_name_y, data_folder_path)
    
    X_scaler_file_path = model_meta['files']['XScalerFile']["filePath"]
    y_scaler_file_path = model_meta['files']['yScalerFile']["filePath"]
    model_file_path = model_meta['files']['modelFile']["filePath"]

    feature_list = model_meta["featureList"]
    target = model_meta["target"]
    scaler_param = model_meta["scalerParam"]
    model_method = model_meta["model_method"]
    train_parameter = model_meta["trainParameter"]

    # Scaling Test Input
    test_X, scaler_X = p4.getScaledTestData(data_X[feature_list], X_scaler_file_path, scaler_param)
    test_y, scaler_y = p4.getScaledTestData(data_y[target], y_scaler_file_path, scaler_param)

    rt = RT()
    rt.set_param(train_parameter)
    rt.set_data(test_X, test_y, window_num)
    model = model_manager.load_pickle_model(model_file_path)
    preds, trues, mse, mae = rt.get_result(model)

    df_result = p4.getPredictionDFResult(preds, trues, scaler_param, scaler_y, featureList= target, target_col = target[0])
    result_metrics =  metrics.calculate_metrics_df(df_result)

    return df_result, result_metrics




# Inference
def get_inference_result(data_X, model_meta, window_num=0, db_client=None):
    
    X_scaler_file_path = model_meta['files']['XScalerFile']["filePath"]
    y_scaler_file_path = model_meta['files']['yScalerFile']["filePath"]
    model_file_path = model_meta['files']['modelFile']["filePath"]

    feature_list = model_meta["featureList"]
    target = model_meta["target"]
    scaler_param = model_meta["scalerParam"]
    model_method = model_meta["model_method"]
    train_parameter = model_meta["trainParameter"]

    print(feature_list)
    print(target)

    # Scaling Test Input
    test_X, scaler_X = p4.getScaledTestData(data_X[feature_list], X_scaler_file_path, scaler_param)
    sacler_y = p4.getScalerFromFile(y_scaler_file_path)

    ri = RI()
    ri.set_param(train_parameter)
    ri.set_data(test_X, window_num)
    model = model_manager.load_pickle_model(model_file_path)
    preds = ri.get_result(model)

    if scaler_param =='scale':
        baseDFforInverse = pd.DataFrame(columns=target, index=range(len(preds)))
        baseDFforInverse[target] = preds
        prediction_result = pd.DataFrame(sacler_y.inverse_transform(baseDFforInverse), columns=target, index=baseDFforInverse.index)
    else:
        prediction_result = pd.DataFrame(columns=target, index=range(len(preds)))

    return prediction_result