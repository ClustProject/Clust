import sys
import pandas as pd
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from Clust.clust.transformation.purpose import machineLearning as ML
from Clust.clust.ML.tool.scaler import get_scaled_data


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


def get_train_val_data(data, feature_list, scaler_root_path, split_ratio, scaler_param, scaler_method ='minmax', mode = None, windows=None):
    train_val, scaler_file_path = get_scaled_data(scaler_param, scaler_root_path, data[feature_list], scaler_method)
    train, val = ML.split_data_by_ratio(train_val, split_ratio, mode, windows)
    
    return train, val, scaler_file_path