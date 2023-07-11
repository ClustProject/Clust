import sys
sys.path.append("../..")
sys.path.append("../../../")

import math
import datetime
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

from Clust.clust.meta.metaDataManager import bucketMeta
from Clust.app import airquality_clustering

def set_pipeline_preprocessing_param(processing_freq, feature_name, bucket_name = None, mongo_client = None, uncertain_nan = None):
    param_dict = {}
    if bucket_name:
        min_max = bucketMeta.get_min_max_info_from_bucketMeta(mongo_client, bucket_name)
        outlier_param ={
            "certain_error_to_NaN": {'flag': True, 'data_min_max_limit':min_max}, 
            "uncertain_error_to_NaN" : {'flag': False}}
        if uncertain_nan:
            ## pipeline의 set_outlier_param 함수 input 에 맞춰서 넣기
            #un_certain = {"algorithm": "SD", "percentile":95, "alg_parameter":{"period":7, "limit":5}}
            un_certain = {'algorithm': 'SR', 'percentile': 95, 'alg_parameter' : {'period': 144}}
            
            outlier_param['uncertain_error_to_NaN']['flag'] = True
            outlier_param['uncertain_error_to_NaN']["outlier_detector_config"] = un_certain
        
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
        "quality_param":{"nan_processing_param":{'type':'num', 'ConsecutiveNanLimit':4, 'totalNaNLimit':18}}
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
    uncertain_nan = pipeline_case_param["uncertain_nan"]

    if (pipeline_case == "case_11") or (pipeline_case == "case_12"):
        pipeline = air_case_1_1_case_1_2(bucket_name, processing_freq, feature_name, mongo_client, uncertain_nan)

    elif pipeline_case == "case_13":
        pipeline = air_case_1_3(bucket_name, processing_freq, feature_name, mongo_client, uncertain_nan)

    elif pipeline_case == "case_14":
        pipeline = air_case_1_4(processing_freq, feature_name)

    return pipeline

def air_case_1_1_case_1_2(bucket_name, processing_freq, feature_name, mongo_client, uncertain_nan):
    param = set_pipeline_preprocessing_param(processing_freq, feature_name, bucket_name, mongo_client, uncertain_nan)

    pipeline = [['data_refinement', param["refine_param"]],
                ['data_outlier', param["outlier_param"]],
                ['data_split', param["split_param"]],
                ['data_integration', param["integration_param"]],
                ['data_quality_check', param["quality_param"]]]
    
    return pipeline

def air_case_1_3(bucket_name, processing_freq, feature_name, mongo_client, uncertain_nan):
    param = set_pipeline_preprocessing_param(processing_freq, feature_name, bucket_name, mongo_client, uncertain_nan)

    pipeline = [['data_refinement', param["refine_param"]],
                ['data_outlier', param["outlier_param"]],
                ['data_split', param["split_param"]],
                ['data_integration', param["integration_param"]]]
    return pipeline

def air_case_1_4(processing_freq, feature_name):
    param = set_pipeline_preprocessing_param(processing_freq, feature_name)

    pipeline = [['data_refinement', param["refine_param"]],
                ['data_integration', param["integration_param"]]]
    
    return pipeline

def get_univariate_df_by_integrating_vertical(processing_data, start_time, feature, frequency):
    """
    입력 DataSet 혹은 DataFrame(Horizontal integration)을 하나의 Feature만 갖는 데이터로 변형하는 모듈
    즉, DataSet의 Data들 혹은 Horizontal integration data의 각 열들을 vertical integration을 하여 하나의 Feature만 갖는 데이터로 변형

    Args:
        processing_data (Dictionary or Dataframe) : 입력 Data set 혹은 Horizontal integration data
        start_time (pd.to_datetime) : 데이터 통합시 새로 설정하고 싶은 시작 시간
        feature (string) : 통합된 데이터의 컬럼 명
        frequency (int) : 데이터 통합시 새로 설정하고 싶은 freq

    Return:
        DataFrame (result_df) : univariate df
    """
    result_df = pd.DataFrame()
    for name in processing_data:
        result_df = pd.concat([result_df, processing_data[name]])
    result_df.columns = [feature]
    time_index = pd.date_range(start=start_time, freq = str(frequency)+"T", periods=len(result_df))
    result_df.set_index(time_index, inplace = True)
    
    return result_df

def clustering_interface(pipeline_case, processing_data, cluster_num, cluster_result_name = None):
    if pipeline_case == "case_11":
        ## clustering 시작
        clustering_result, clustering_result_by_class = airquality_clustering.pipeline_clustering(processing_data, cluster_num)
        print(clustering_result_by_class)
        
        clustering_result_json_name = "../../../{}.json".format(cluster_result_name)
        with open(clustering_result_json_name, 'w') as f:
            json.dump(clustering_result, f)

    elif (pipeline_case == "case_12") or (pipeline_case == "case_13") or (pipeline_case == "case_14"):
        clustering_result = None

    return clustering_result

def get_univariate_df(pipeline_case, processing_data, new_start_time, feature_name, processing_freq, clustering_result = None, select_class = None):
    if pipeline_case == "case_11":
        ## 선택한 clust class 기반으로 integration vertical 하여 새로운 데이터 생성
        result_df = airquality_clustering.get_univariate_df_by_selecting_clustering_data(processing_data, clustering_result, new_start_time, feature_name, processing_freq, select_class)
        result_df.plot()

    elif (pipeline_case == "case_12") or (pipeline_case == "case_13") or (pipeline_case == "case_14"):
        result_df = get_univariate_df_by_integrating_vertical(processing_data, new_start_time, feature_name, processing_freq)
        result_df.plot()
    
    return result_df