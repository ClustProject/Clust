import sys
sys.path.append("../..")
sys.path.append("../../../")

import math
import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from Clust.clust.pipeline import data_pipeline
from Clust.clust.meta.metaDataManager import bucketMeta

def set_pipeline_preprocessing_param(processing_freq, feature_name, bucket_name = None, mongo_client = None):
    param_dict = {}
    if bucket_name:
        min_max = bucketMeta.get_min_max_info_from_bucketMeta(mongo_client, bucket_name)
        outlier_param ={
            "certain_error_to_NaN": {'flag': True, 'data_min_max_limit':min_max}, 
            "uncertain_error_to_NaN":{'flag': False}}
        param_dict["outlier_param"] = outlier_param
    
    timedelta_frequency_min = datetime.timedelta(minutes= processing_freq)

    refine_param = {"remove_duplication": {'flag': True}, 
                    "static_frequency": {'flag': True, 'frequency': timedelta_frequency_min}}
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
    quality_param = {
        "quality_method":"data_with_clean_feature", 
        "quality_param":{"nan_processing_param":{'type':'num', 'ConsecutiveNanLimit':4, 'totalNaNLimit':19}}
    }
    param_dict["refine_param"] = refine_param
    param_dict["split_param"] = cycle_split_param
    param_dict["integration_param"] = integration_param
    param_dict["quality_param"] = quality_param

    return param_dict

def get_pipeline(pipeline_case, pipeline_case_param):
    """
    공기질 시나리오의 Processing Data 단계를 pipeline case 별 진행하는 interface

    Example:
        >>> integration_freq_min = 10
        >>> feature_name = 'in_co2'

        >>> pipeline_case_param = {
        ...    "bucket_name" : bucket,
        ...    "processing_freq" : integration_freq_min,
        ...    "feature_name" : feature_name,
        ...    "mongo_client" : mongo_client_
        ... }
    """
    bucket_name = pipeline_case_param["bucket_name"]
    processing_freq = pipeline_case_param["processing_freq"]
    feature_name = pipeline_case_param["feature_name"]
    mongo_client = pipeline_case_param["mongo_client"]

    if pipeline_case == "case_11":
        pipeline = air_case_1_1_case_1_2(bucket_name, processing_freq, feature_name, mongo_client)

    elif pipeline_case == "case_12":
        pipeline = air_case_1_1_case_1_2(bucket_name, processing_freq, feature_name, mongo_client)

    elif pipeline_case == "case_13":
        pipeline = air_case_1_3(bucket_name, processing_freq, feature_name, mongo_client)

    elif pipeline_case == "case_14":
        pipeline = air_case_1_4(processing_freq, feature_name)

    return pipeline

def air_case_1_1_case_1_2(bucket_name, processing_freq, feature_name, mongo_client):
    param = set_pipeline_preprocessing_param(processing_freq, feature_name, bucket_name, mongo_client)

    pipeline = [['data_refinement', param["refine_param"]],
                ['data_outlier', param["outlier_param"]],
                ['data_split', param["cycle_split_param"]],
                ['data_integration', param["integration_param"]],
                ['data_quality_check', param["quality_param"]]]
    
    return pipeline

def air_case_1_3(bucket_name, processing_freq, feature_name, mongo_client):
    param = set_pipeline_preprocessing_param(processing_freq, feature_name, bucket_name, mongo_client)

    pipeline = [['data_refinement', param["refine_param"]],
                ['data_outlier', param["outlier_param"]],
                ['data_split', param["cycle_split_param"]],
                ['data_integration', param["integration_param"]]]
    return pipeline

def air_case_1_4(processing_freq, feature_name):
    param = set_pipeline_preprocessing_param(processing_freq, feature_name)

    pipeline = [['data_refinement', param["refine_param"]],
                ['data_integration', param["integration_param"]]]
    
    return pipeline

def get_univariate_df_by_integrating_vertical(processing_data, start_time, feature, frequency):
    result_df = pd.DataFrame()
    for name in processing_data:
        result_df = pd.concat([result_df, processing_data[name]])
    result_df.columns = [feature]
    time_index = pd.date_range(start=start_time, freq = str(frequency)+"T", periods=len(result_df))
    result_df.set_index(time_index, inplace = True)
    
    return result_df