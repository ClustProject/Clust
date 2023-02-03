import sys
sys.path.append("../")
sys.path.append("../../")

from Clust.clust.ML.common.common import p3_training as p3
import pandas as pd 

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
        scaledD = pd.DataFrame(scaler.transform(data), index = data.index, columns = data.columns)
    else:
        scaledD = data.copy()
    return scaledD

def get_prediction_df_result(predictions, values, scaler_param, scaler, feature_list, target_col):
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


def get_cleand_data(data, clean_train_data_param, integration_freq_sec, nan_processing_param):
    if clean_train_data_param =='Clean':
        import datetime
        timedelta_frequency_sec = datetime.timedelta(seconds= integration_freq_sec)
        result = p3.clean_nan_df(data, nan_processing_param,  timedelta_frequency_sec)

    else:
        result = data.copy()
        pass
    
    return result