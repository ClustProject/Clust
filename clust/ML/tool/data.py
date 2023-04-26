import sys, os
import pandas as pd
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from Clust.clust.transformation.purpose import machineLearning as ML
from Clust.clust.ML.tool import scaler

# p2_dataSelection
def get_saved_integrated_data(data_save_mode, data_name, data_folder_path=None, db_name=None, db_client = None):
    """
    if data_save_mode == CSV, get data from csv file

    if data_save_mode == influx, get data from influx db

    Args:
        data_save_mode (string): data save mode
        data_name (string): integrated data name
        data_folder_path (string): folder path
        db_name (string): data name
        db_client (db_client): db_client

    Returns:
        data (dataframe) : data
    
    """
    if data_save_mode =='CSV':
        file_name = os.path.join(data_folder_path, data_name +'.csv')
        try:
            data = pd.read_csv(file_name, index_col='datetime', infer_datetime_format=True, parse_dates=['datetime'])
        except ValueError:
            data = pd.read_csv(file_name, index_col='Unnamed: 0', infer_datetime_format=True, parse_dates=['Unnamed: 0'])

    elif data_save_mode =='influx':
        ms_name = data_name
        data = db_client.get_data(db_name, ms_name)
        
    return data


def DF_to_series(data):
    """make input data for clustering. 
    Args:
        data(np.dataFrame): input data
    Return:
        series_data(series): transformed data for training, and prediction

    """
    series_data = data.to_numpy().transpose()
    return series_data



def get_prediction_df_result(predictions, values, scaler_param, scaler, feature_list, target_col):
    """
    if scaler_param == scale, inverse scaled prediction data

    Args:
        predictions (ndarray): prediction data
        values (ndarray): true data
        scaler_param (string): scale or noscale
        scaler (scaler): scaler name
        feature_list (list): feature list
        target_col (string): target column

    Returns:
        df_result (dataframe) : prediction & true data
    
    """
    print(scaler_param)
    if scaler_param =='scale':
        base_df_for_inverse = pd.DataFrame(columns=feature_list, index=range(len(predictions)))
        base_df_for_inverse[target_col] = predictions
        prediction_inverse = pd.DataFrame(scaler.inverse_transform(base_df_for_inverse), columns=feature_list, index=base_df_for_inverse.index)
        base_df_for_inverse[target_col] = values 
        values_inverse = pd.DataFrame(scaler.inverse_transform(base_df_for_inverse), columns=feature_list, index=base_df_for_inverse.index)
        trues = values_inverse[target_col]
        preds = prediction_inverse[target_col]
        df_result = pd.DataFrame(data={"value": trues, "prediction": preds}, index=base_df_for_inverse.index)

    else:
        df_result = pd.DataFrame(data={"value": values, "prediction": predictions}, index=range(len(predictions)))
    
    return df_result

# p3_training
def get_train_val_data(data, feature_list, scaler_root_path, split_ratio, scaler_param, scaler_method ='minmax', mode = None, windows=None):
    """
    scaled data, split data by ratio

    Args:
        data (dataframe): data
        feature_list (list): feature list
        scaler_root_path (string):scaler file path
        split_ratio (int): split ratio
        scaler_param (string): scale or noscale
        scaler_method (string): minmax scaler
        mode (string): Classification or windows_split or others
        windows (int): window size

    Returns:
        train (dataframe) : train data
        val (dataframe) : validation data
        scaler_file_path (string) : scaler file path
    
    """
    train_val, scaler_file_path = scaler.get_data_scaler(scaler_param, scaler_root_path, data[feature_list], scaler_method)
    train, val = ML.split_data_by_ratio(train_val, split_ratio, mode, windows)
    
    return train, val, scaler_file_path