import sys
sys.path.append("../..")
sys.path.append("../../../")
import pandas as pd
import datetime

from Clust.clust.pipeline import data_pipeline
def get_DFSet_preprocessing_test_pipeline(pipeline_case_param, uncertain_error_to_NaN_flag, param):
    """
    Data Set input과 파라미터 값을 의거하여 유효한 테스트 파이프라인 파라미터를 생성함
    Args:
        preprocessing_case_param(dict)
        uncertain_error_to_NaN_flag (str)
        param (str)test_pipe_param

        
    Example:
        >>> processing_case_param = {
        ... "processing_task_list" : processing_task_list, 
        ... "data_min_max" : data_min_max,
        ... "processing_freq" : processing_freq,
        ... "feature_name" : feature_name,
        ... }
    """
    data_min_max = pipeline_case_param["data_min_max"]
    processing_freq = pipeline_case_param["processing_freq"]
    feature_name = pipeline_case_param["feature_name"]
    timedelta_frequency_min = datetime.timedelta(minutes= processing_freq)
    
    
    processing_task_list = pipeline_case_param["processing_task_list"]

    print("featureName", feature_name)
    param['data_refinement']['static_frequency']['frequency'] = timedelta_frequency_min
    param['data_integration']['integration_param']['integration_frequency'] = timedelta_frequency_min
    param['data_integration']['integration_param']['feature_name'] = feature_name
    param['data_outlier']['certain_error_to_NaN']['data_min_max_limit'] = data_min_max
    param['data_outlier']['uncertain_error_to_NaN']['flag'] = uncertain_error_to_NaN_flag
        
    pipeline = []
    for procssing_task in processing_task_list:
        pipeline.append([procssing_task, param[procssing_task]])
    valid_flag = data_pipeline.pipeline_connection_check(pipeline, input_type = 'DFSet')
    
    if valid_flag:
        return pipeline
    else:
        print("It's not working")
        return None

############# Clustering






