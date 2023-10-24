import sys
import pandas as pd
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from Clust.clust.transformation.general.dataScaler import DataScaler

 # p4_testing
def get_scaled_test_data(data, scaler_file_path, scaler_param):
    """
    if scaler_param == scale, get scaler & data scaling

    Args:
        data (dataframe): train data
        scaler_file_path (string): clean or noclean
        scaler_param (int): scale or noscale

    Returns:
        result (dataframe) : scaled data
        scaler (scaler) : scaler
    
    """
    scaler =None
    result = data
    if scaler_param =='scale':
        if scaler_file_path:
            scaler = get_scaler_file(scaler_file_path)
            result = get_scaled_data(data, scaler, scaler_param)
    return result, scaler

 # p4_testing
def get_scaler_file(scaler_file_path):
    """
    get scaler file

    Args:
        scaler_file_path (string): clean or noclean

    Returns:
        scaler (scaler) : scaler    
    
    """
    import joblib
    scaler = joblib.load(scaler_file_path)
    return scaler

 # p4_testing
def get_scaled_data(data, scaler, scaler_param):
    """
    if scaler_param == scale, data scaling

    Args:
        data (dataframe): train data
        scaler (scaler): scaler
        scaler_param (int): scale or noscale

    Returns:
        scaled_data (dataframe) : scaled data
    
    """
    if scaler_param=='scale':
        scaled_data = pd.DataFrame(scaler.transform(data), index = data.index, columns = data.columns)
    else:
        scaled_data = data.copy()
    return scaled_data


# p3_training
def get_data_scaler(scaler_param, scaler_root_path, data, scaler_method):
    """
    data scaling

    """
    if scaler_param=='scale':
        DS = DataScaler(scaler_method, scaler_root_path)
        #from Clust.clust.transformation.general import dataScaler
        #feature_col_list = dataScaler.get_scalable_columns(train_o)
        DS.setScaleColumns(list(data.columns))
        DS.setNewScaler(data)
        result_data = DS.transform(data)
        scaler_file_path = DS.scalerFilePath
        
    else:
        result_data = data.copy()
        scaler_file_path=None

    return result_data, scaler_file_path
