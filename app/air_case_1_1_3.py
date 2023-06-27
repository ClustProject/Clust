import sys
sys.path.append("../../../")
sys.path.append("../..")

import math
import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from Clust.clust.pipeline import data_pipeline
from Clust.clust.ML.clustering.interface import clusteringByMethod
from Clust.clust.tool.plot import plot_interface
from Clust.clust.ML.tool import util

def set_case_113_pipeparam(bucket, integration_freq_min, feature_name):
    timedelta_frequency_min = datetime.timedelta(minutes= integration_freq_min)
    integration_param={
        "integration_param":{"feature_name":feature_name, "duration":None, "integration_frequency":timedelta_frequency_min},
        "integration_type":"one_feature_based_integration"
    }

    pipeline = [['data_integration', integration_param]]
    
    return pipeline

def get_univariate_df_by_integrating_vertical(processing_data, start_time, feature, frequency):
    result_df = pd.DataFrame()
    for name in processing_data:
        result_df = pd.concat([result_df, processing_data[name]])
    result_df.columns = [feature]
    time_index = pd.date_range(start=start_time, freq = str(frequency)+"T", periods=len(result_df))
    result_df.set_index(time_index, inplace = True)
    
    return result_df

def get_train_test_data_by_days(data, freq):
    data_day_length = int(len(data)/(24*60/freq))
    print("data_day_length : ", data_day_length)
    split_day_length = int(data_day_length*0.8)
    print("train_day_length : ", split_day_length)
    split_length = int(split_day_length*(24*60/freq))

    train = data[:split_length]
    test = data[split_length:]

    return train, test