import pandas as pd
import sys
sys.path.append("../")

from Clust.clust.tool.stats_table import metrics
from Clust.clust.ML.common import model_manager
from Clust.clust.ML.common.common import p2_dataSelection as p2
from Clust.clust.ML.common.common import p4_testing as p4
from Clust.clust.ML.forecasting.test import ForecasatingTest as FT
from Clust.clust.ML.forecasting.inference import ForecastingInfernce as FI


def get_test_result(data_name, data_meta, model_meta, data_folder_name, db_client=None):

    data_save_mode = data_meta[data_name]["integrationInfo"]["DataSaveMode"]
    data = p2.getSavedIntegratedData(data_save_mode, data_name, data_folder_name)
    
    scaler_file_path = model_meta['files']['scalerFile']["filePath"]
    model_file_path = model_meta['files']['modelFile']["filePath"]

    feature_list = model_meta["featureList"]
    target_col = model_meta['transformParameter']["target_col"]
    scaler_param = model_meta["scalerParam"]
    model_method = model_meta["model_method"]
    train_parameter = model_meta["trainParameter"]
    transform_parameter = model_meta["transformParameter"]
    integration_freq_sec = model_meta['trainDataInfo']["integration_freq_sec"]
    clean_train_data_param = model_meta["cleanTrainDataParam"] 
    nan_processing_param = model_meta['NaNProcessingParam']

    test_data, scaler = p4.getScaledTestData(data[feature_list], scaler_file_path, scaler_param)
    clean_test_data = p4.getCleandData(test_data, clean_train_data_param, integration_freq_sec, nan_processing_param)

    ft = FT()
    ft.set_param(model_meta)
    ft.set_data(clean_test_data)
    model = model_manager.load_pickle_model(model_file_path)
    preds, trues = ft.get_result(model)

    df_result = p4.getPredictionDFResult(preds, trues, scaler_param, scaler, feature_list, target_col)
    df_result.index = test_data[(transform_parameter['future_step']+transform_parameter['past_step']-1):].index
    result_metrics =  metrics.calculate_metrics_df(df_result)

    return df_result, result_metrics




def get_inference_result(data, model_meta):

    scaler_file_path = model_meta['files']['scalerFile']["filePath"]
    model_file_path = model_meta['files']['modelFile']["filePath"]

    feature_list = model_meta["featureList"]
    target_col = model_meta['transformParameter']["target_col"]
    scaler_param = model_meta["scalerParam"]
    past_step = model_meta['transformParameter']['past_step']

    feature_data = data[feature_list]
    step_data = feature_data[-past_step:][feature_list].values
    df_data = pd.DataFrame(step_data, columns = feature_list)

    input_data, scaler = p4.getScaledTestData(df_data[feature_list], scaler_file_path, scaler_param)

    fi = FI()
    fi.set_param(model_meta)
    fi.set_data(input_data)
    model = model_manager.load_pickle_model(model_file_path)
    preds = fi.get_result(model)


    if scaler_param =='scale':
        base_df_for_inverse= pd.DataFrame(columns=feature_list, index=range(len(preds)))
        base_df_for_inverse[target_col] = preds
        inverse_result = pd.DataFrame(scaler.inverse_transform(base_df_for_inverse), columns=feature_list, index=base_df_for_inverse.index)
        target_data = inverse_result[target_col]
        prediction_result = pd.DataFrame(data={target_col: target_data}, index=range(len(preds)))
        
    else:
        prediction_result = pd.DataFrame(data={target_col: preds}, index=range(len(preds)))


    return prediction_result
