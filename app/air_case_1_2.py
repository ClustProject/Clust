import sys
sys.path.append("../../../")
sys.path.append("../..")

import math
import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from Clust.clust.pipeline import data_pipeline
from Clust.setting import influx_setting_KETI as ins
from Clust.clust.ingestion.mongo import mongo_client
from Clust.clust.meta.metaDataManager import bucketMeta

from Clust.clust.ML.clustering.interface import clusteringByMethod
from Clust.clust.tool.plot import plot_interface
from Clust.clust.ML.tool import util

def set_case_112_pipeparam(bucket, integration_freq_min, feature_name):
    # 1. refine_param
    mongo_client_ = mongo_client.MongoClient(ins.CLUSTMetaInfo2)
    min_max = bucketMeta.get_min_max_info_from_bucketMeta(mongo_client_, bucket)
    timedelta_frequency_min = datetime.timedelta(minutes= integration_freq_min)

    feature_name = 'in_co2'

    refine_param = {"remove_duplication": {'flag': True}, 
                    "static_frequency": {'flag': True, 'frequency': timedelta_frequency_min}}

    outlier_param ={
        "certain_error_to_NaN": {'flag': True, 'data_min_max_limit':min_max}, 
        "uncertain_error_to_NaN":{'flag': False}}


    cycle_split_param={
        "split_method":"cycle",
        "split_param":{
            'feature_cycle' : 'Day',
            'feature_cycle_times' : 1}
    }

    integration_param={
        "integration_param":{"feature_name":feature_name, "duration":None, "integration_frequency":timedelta_frequency_min },
        "integration_type":"one_feature_based_integration"
    }

    pipeline = [['data_refinement', refine_param],
                ['data_outlier', outlier_param],
                ['data_split', cycle_split_param],
                ['data_integration', integration_param]]
    
    return pipeline

def get_univariate_df_by_integrating_vertical(processing_data, start_time, feature, frequency):
    result_df = pd.DataFrame()
    for name in processing_data:
        result_df = pd.concat([result_df, processing_data[name]])
    result_df.columns = [feature]
    time_index = pd.date_range(start=start_time, freq = str(frequency)+"T", periods=len(result_df))
    result_df.set_index(time_index, inplace = True)
    
    return result_df
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