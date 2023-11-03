import sys
import pandas as pd
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")

# from sklearn.metrics import classification_report
from Clust.clust.tool.stats_table import metrics
from sklearn.metrics import roc_auc_score, f1_score
from Clust.clust.ML.tool import scaler as ml_scaler
from Clust.clust.ML.tool import data as ml_data
from Clust.clust.ML.common import AD_pipeline
from Clust.clust.ML.common import echart
import numpy as np

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")

def convert_param_for_backend(params):
    """
    frontend에서 전달하는 파라미터를 백엔드에서 원활하게 사용하고, 필요한 데이터를 추가하여 백엔드로 보내기 위함

    - 향후 서비스에 맞게 수정해야함

    Args:
        params (dict): dictionary data input

    Returns:
        dictionary : params(dictionary data output)

    """
    # chage tpye string to bool -> ex) 'true' -> True
    params = chagne_type_str_to_bool(params)
    
    # model name check  
    params['model_info']['model_name'] = check_model_name(params['model_info']['model_name'],
                                        [params['ingestion_param_X']['ms_name'], params['model_info']['model_purpose'], params['model_info']['model_method']])
    
    # system 
    params['model_info']['train_parameter']['device'] = device
    
    # float to integer
    if 'past_step' in list(params['transform_param'].keys()):
        params['transform_param']['past_step'] = int(params['transform_param']['past_step'])
        params['transform_param']['future_step'] = int(params['transform_param']['future_step'])
    
    
    # hard parameter setting
    if params['transform_param']['data_clean_option']:
        params['transform_param']['nan_process_info']= {'type':'num', 'ConsecutiveNanLimit':10, 'totalNaNLimit':100}
        params['transform_param']['max_nan_limit_ratio'] = 0.9
    else:
        params['transform_param']['nan_process_info'] = {'type':'num', 'ConsecutiveNanLimit':10000, 'totalNaNLimit':100000}
        params['transform_param']['max_nan_limit_ratio'] = 0.5
    return params

def chagne_type_str_to_bool(dict_data):
    """
    frontend에서 전달하는 dictionary json data를 python에서 활용 가능한 형태로 bool string을 변형함

    Args:
        dict_data (dict): dictionary data input

    Returns:
        dictionary : dict_data(dictionary data output)
    """

    for key, value in dict_data.items():

        if isinstance(value, dict):
            dict_data[key] = chagne_type_str_to_bool(value)

        elif isinstance(value, str):

            if value.lower() == 'true':
                dict_data[key] = True
            elif value.lower() == 'false':
                dict_data[key] = False
            elif value.lower() == 'none':
                dict_data[key] = None
            elif value.lower() == 'nan':
                dict_data[key] = np.nan

    return dict_data

def check_model_name(model_name, model_name_info):
    """
    It makes model name by default value and additional information

    Args:
        model_name (string): default model name
        model_name_info (array): model name information

    Returns:
        string : model_name(final model name)
    """
    # model name & path
    if model_name is None or model_name == 'None':
        model_name=""
        for key in model_name_info:
            model_name+=key+'_'
        
    return model_name

# # --------------------------------- training ---------------------------------------------------

# TODO:
def train_data_preparation(params):
    """
    prepare data for train using AD_pipeline

    Args:
        params (dict): parameters for data preparation

    Returns:
        train_dataset (torch.Tensor): train dataset
        test_dataset (torch.Tensor): test dataset
    """

    pass

def AD_train(params, train_X, train_y, val_X, val_y):
    """ 
    rain using train/test data and return model information

    Args:
        params (dict): parameters including 'model_info'.
        train_X (DataFrame): train data
        test_X (DataFrame): test data

    Returns:
        dictionary: params(trained model info including model file path)
    """
    # model info update (if necessary)
    from Clust.clust.ML.common import model_parameter_setting
    params['model_info']['seq_len'] = train_X.shape[0]  # TODO: batch? 
    params['model_info']['input_size'] = train_X.shape[1]  # TODO: batch?
    params['model_info']['model_parameter'] = model_parameter_setting.set_model_parameter(params['model_info']) 
    
    from Clust.clust.ML.tool import model as ml_model
    train_data_path_list = [params['model_info']['model_name'], params['ingestion_param_X']['ms_name']]
    # train_data_path_list = [params['model_info']['model_name'], params['data_param_X']['name']]
    model_file_path = ml_model.get_model_file_path(train_data_path_list, params['model_info']['model_method'])

    params['model_info']['model_file_path'] = {
        'modelFile':{
            'fileName': 'model.pth',
            'filePath': model_file_path
        }
    }

    # model training
    if params['model_info']['model_purpose'] == 'anomaly_detection':
        AD_pipeline.CLUST_anomalyDet_train(train_X,
                                            train_y,
                                            val_X,
                                            val_y,
                                            params['model_info'])

    return params

# --------------------------------- test ---------------------------------------------------

# TODO:
def test_data_preparation(params):
    """
    prepare data for test using ML_pipeline
    1. Ingest 
    2. Scale 
    3. Transform

    Args:
        params (dict): parameters including 'ingestion_param_X', 'ingestion_param_y', 'scaler_param', and 'transform_param'.
        influxdb_client (influxdb client): influxdb client.

    Returns:
        np.array: test_X_array, test_y_array

    Returns:
        scaler: scaler_X, scaler_y
    """

    pass

def AD_test(params, test_X, test_y, scaler):
    """
    test using given data and model information

    Args:
        params (dict): parameters including 'model_info', 'scaler_param', 'ingestion_param_X' and 'ingestion_param_y'.
        test_X (DataFrame): test X data
        test_y (DataFrame): test y data
        scaler (scaler): X or y sclaer

    Returns:
        dictionary : result(dictionary contains Echart format Dataframe result and result metrics)
    """
    if params['data_y_flag']:
        feature_list = params['ingestion_param_y']['feature_list']
    else:
        feature_list = params['ingestion_param_X']['feature_list']
    target = params['ingestion_param_y']['feature_list'][0]
    
    result_metrics = dict()
    if params['model_info']['model_purpose'] == 'anomaly_detection':
        preds, trues, thres = AD_pipeline.CLUST_anomalyDet_test(test_X,
                                                         test_y,
                                                         params['model_info'])
        df_result = ml_data.get_prediction_df_result(preds, trues, scaler_param=False, scaler=None, feature_list=feature_list, target_col=target)
        result_metrics = metrics.calculate_anomaly_metrics(preds, trues, thres)

    result = {'result': echart.getEChartFormatResult(df_result), 'result_metrics': result_metrics}

    return result

# --------------------------------- inference ---------------------------------------------------
def _get_scaled_np_data(data, scaler, scaler_param):
    """
    scaling given data in numpy array format

    Args:
        data (ndarray): given data
        scaler (scaler): scaler for given data
        scaler_param (bool): scaler flag

    Returns:
        ndarray : scaled_data
    """
    if scaler_param=='scale':
        scaled_data = scaler.transform(data)
    else:
        scaled_data = data.copy()
    return scaled_data

def _get_scaled_infer_data(data, scaler_file_path, scaler_param):
    """
    return scaled data and scaler

    Args:
        data (ndarray): given data
        scaler_file_path (str): scaler file path
        scaler_param (bool): scaler flag

    Returns:
        ndarray : result(scaled data)

    Returns:
        scaler : scaler

    """
    scaler =None
    result = data
    if scaler_param =='scale':
        if scaler_file_path:
            scaler = ml_scaler.get_scaler_file(scaler_file_path)
            result = _get_scaled_np_data(data, scaler, scaler_param)
    return result, scaler

def infer_data_preparation(params, data):
    """
    return scaled X data and y scaler
    1. Ingest 
    2. Scale
        
    Args:
        params (dict):
        data (ndarray): given data X for inference

    Returns:
        ndarray : scaled_infer_X(scaled data X)

    Returns:
        scaler : scaler_y

    """
    
    scaled_infer_X, scaler_X = _get_scaled_infer_data(data, params['scaler_param']['scaler_file_path']['XScalerFile']["filePath"], params['scaler_param']['scaler_flag'])
    scaler_y = ml_scaler.get_scaler_file(params['scaler_param']['scaler_file_path']['yScalerFile']["filePath"])
    
    return scaled_infer_X, scaler_y

def AD_inference(params, infer_X, scaler):
    """
    _summary_

    Args:
        params (dict): parameters including 'model_info', 'scaler_param', and 'ingestion_param_y'
        infer_X (DataFrame): inference X data
        scaler (scaler): y scaler

    Returns:
        pd.DataFrame : prediction_result

    """
    target = params['ingestion_param_y']['feature_list']

    if params['model_info']['model_purpose'] == 'anomaly_detection':
        preds = AD_pipeline.CLUST_anomalyDet_inference(infer_X, params['model_info'])

    if params['scaler_param']['scaler_flag'] =='scale':
        base_df_for_inverse = pd.DataFrame(columns=target, index=range(len(preds)))
        base_df_for_inverse[target[0]] = preds
        prediction_result = pd.DataFrame(scaler.inverse_transform(base_df_for_inverse), columns=target, index=base_df_for_inverse.index)
    else:
        prediction_result = pd.DataFrame(data={"value": preds}, index=range(len(preds)))
            
    return prediction_result