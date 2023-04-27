import sys, os
sys.path.append("../")
sys.path.append("../../")
import pandas as pd
def get_Xy_data_in_Clust (ingestion_param_X, data_y_flag, ingestion_param_y, ingestion_method, db_client):
    """_summary_

    Args:
        ingestion_param_X (_type_): _description_
        data_y_flag (_type_): _description_
        ingestion_param_y (_type_): _description_
        ingestion_method (_type_): _description_
        db_client (_type_): _description_

    Returns:
        _type_: _description_
    """
    from Clust.clust.data import data_interface
    data_X = data_interface.get_data_result(ingestion_method, db_client, ingestion_param_X)
    if data_y_flag:
        data_y = data_interface.get_data_result(ingestion_method, db_client, ingestion_param_y)

    else: # y 가 없다면 데이터를 만들어야 함
        feature_y_list = ingestion_param_y['feature_list']
        data_y = data_X[feature_y_list]
        
    return data_X, data_y

def random_nan_df(df, nan_ratio):
    for col in df.columns:
        df.loc[df.sample(frac=nan_ratio). index, col] = pd.np.nan
    return df

def get_scaled_data_in_Clust(data_name_X, data_X, data_name_y, data_y, scaler_path, scaler_param, scale_method):
    scalerRootPath_X = os.path.join(scaler_path, data_name_X)

    # X Data Scaling
    from Clust.clust.ML.tool import scaler
    scalerRootPath_X = os.path.join(scaler_path, data_name_X)
    dataX_scaled, X_scalerFilePath = scaler.get_data_scaler(scaler_param, scalerRootPath_X, data_X, scale_method)   
    
    # X Data Scaling
    scalerRootPath_y = os.path.join(scaler_path, data_name_y)
    datay_scaled, y_scalerFilePath = scaler.get_data_scaler(scaler_param, scalerRootPath_y, data_y, scale_method)
    
    return dataX_scaled, X_scalerFilePath, datay_scaled, y_scalerFilePath

def get_clean_model_data_1(model_clean, nan_process_info, data):
    if model_clean:
        from Clust.clust.quality.NaN import cleanData
        CMS = cleanData.CleanData()
        data = CMS.get_cleanData_by_removing_column(data, nan_process_info) 
    else:
        pass
    return data

def get_default_day_window_size(data):
    from datetime import timedelta 
    # define window size by clust structure
    first_date = data.index[0]
    day_window_size = data.loc[first_date:first_date + timedelta(days =0, hours=23, minutes=59, seconds=59)].shape[0]
    
    return day_window_size
    
from Clust.clust.transformation.purpose import machineLearning as ML
def get_split_data_in_CLUST(split_mode, split_ratio, dataX, datay, day_window_size):
    if split_mode =='window_split':
        train_x, val_x = ML.split_data_by_ratio(dataX, split_ratio, split_mode, day_window_size)
    else:
        train_x, val_x = ML.split_data_by_ratio(dataX, split_ratio, None, None)

    train_y, val_y = ML.split_data_by_ratio(datay, split_ratio, None, None)
    
    return train_x, val_x, train_y, val_y

def get_transfomed_data(split_mode, transformParameter, X, y):
    if split_mode =='window_split':
        from Clust.clust.transformation.type import DFToNPArray
        X_array, y_array= DFToNPArray.trans_DF_to_NP_by_windowNum(X, y, transformParameter)
        
    elif split_mode == 'step_split':
        X_array, y_array = ML.trans_by_step_info(X, y, transformParameter)
        
    return X_array, y_array

def get_default_model_name(model_name, app_name, model_method, model_clean):
    if model_name is None:
        model_name = app_name+ '_'+ model_method + '_' + str(model_clean)
    else:
        pass
    return model_name


def get_default_model_path(model_name,data_name, model_method, train_parameter):
    from Clust.clust.transformation.general.dataScaler import encode_hash_style
    trainParameter_encode =  encode_hash_style(str(train_parameter))
    trainDataPathList = [model_name, data_name, trainParameter_encode]
    from Clust.clust.ML.tool import model as ml_model
    modelFilePath = ml_model.get_model_file_path(trainDataPathList, model_method)
    
    return modelFilePath