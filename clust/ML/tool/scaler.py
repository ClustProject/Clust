import sys
import pandas as pd
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from Clust.clust.transformation.general.dataScaler import DataScaler

# p3.get_scaled_data
def scaled_train_data(scaler_param, scaler_root_path, data, scaler_method):
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


def get_scaled_test_data(data, scaler_file_path, scaler_param):
    scaler =None
    result = data
    if scaler_param =='scale':
        if scaler_file_path:
            scaler = get_scaler_file(scaler_file_path)
            result = get_scaled_data(data, scaler, scaler_param)
    return result, scaler


def get_scaler_file(scaler_file_path):
    import joblib
    scaler = joblib.load(scaler_file_path)
    return scaler


def get_scaled_data(data, scaler, scaler_param):
    if scaler_param=='scale':
        scaled_data = pd.DataFrame(scaler.transform(data), index = data.index, columns = data.columns)
    else:
        scaled_data = data.copy()
    return scaled_data