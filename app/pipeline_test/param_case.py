import sys
sys.path.append("../../../")
sys.path.append("../..")

import pandas as pd
from dateutil.relativedelta import relativedelta

from Clust.clust.meta.metaDataManager import bucketMeta
from Clust.setting import influx_setting_KETI as ins
from Clust.clust.ingestion.mongo import mongo_client
mongo_client_ = mongo_client.MongoClient(ins.CLUSTMetaInfo2)

########################################################################
#### automatically make additional variables
task_name = "air_quality"

def get_processing_task_list(preprocessing_case, data_param):
    if preprocessing_case == "processing_1":
        processing_task_list = ['data_outlier', 'data_refinement', 'data_split', 'data_integration','data_quality_check','data_imputation', 'data_smoothing'] # +clustering

    elif preprocessing_case == "processing_2":
        processing_task_list = [ 'data_outlier', 'data_refinement','data_split', 'data_integration','data_quality_check','data_imputation']

    elif preprocessing_case == "processing_3":
        processing_task_list = ['data_refinement', 'data_outlier', 'data_split', 'data_integration', 'data_imputation']

    elif preprocessing_case == "processing_4":
        test_pipe_param['data_integration']['integration_param']['duration'] = {'start_time': data_param['start_time'], 'end_time': data_param['end_time']}
        processing_task_list = ['data_refinement', 'data_integration', 'data_imputation']

    elif preprocessing_case == "test_data_processing_1":
        processing_task_list = ['data_refinement']
    return processing_task_list

if task_name =='air_quality':
    from . import param_data_airquality as param_data
else:
    pass

def get_test_pipe_param(task_name, pipe_pre_case, cycle_condition):
    test_pipe_param = param_data.get_data_preprocessing_param(pipe_pre_case)

    if cycle_condition == "day_1":
        feature_cycle = 'Day'
        feature_cycle_times = 1 
        
    elif cycle_condition == "week_1":
        feature_cycle = 'Week'
        feature_cycle_times = 1 
        
    test_pipe_param['data_split']['split_param']['feature_cycle'] = feature_cycle
    test_pipe_param['data_split']['split_param']['feature_cycle_times'] = feature_cycle_times
    
    return test_pipe_param

def get_new_ms_name(data_type, cluster_result_name, select_class = None):
    if select_class:
        select_class_str = ''.join(str(c) for c in select_class)
        cluster_result_name = cluster_result_name +"_"+select_class_str
    new_ms_name = cluster_result_name + "_" + data_type

    return new_ms_name


def define_processing_case_param(preprocessing_case, data_level, pipe_pre_case, cycle_condition):
    # [[pipeline_case_num, clustering_flag]] -> [pipeline1-1, pipeline1-2, pipeline1-3, pipeline1-4, test pipeline]

    ########################################################################
    bucket, data_param, processing_freq, feature_name, ingestion_method  = param_data.get_data_conidtion_by_data_level(data_level)
    test_pipe_param = get_test_pipe_param(task_name, pipe_pre_case, cycle_condition)
    min_max = bucketMeta.get_min_max_info_from_bucketMeta(mongo_client_, bucket )
    ########################################################################
    #### preprocessing and clustering method Setting
    
    ## 1. preprocessing param setup
    processing_task_list = get_processing_task_list(preprocessing_case, data_param)
    
    ## 2. preprocessing 
    processing_case_param = {
        "processing_task_list" : processing_task_list, 
        "processing_freq" : processing_freq,
        "feature_name" : feature_name,
        "data_min_max" : min_max
    }

    return processing_case_param, test_pipe_param, ingestion_method, data_param 


## 3. save result 
def get_new_start_time (start_time):
    delta = relativedelta(years=5)
    new_start_time = start_time- delta
    return new_start_time
    
"""
if preprocessing_case == "processing_1":
    processing_task_list = ['data_refinement', 'data_outlier', 'data_split', 'data_integration','data_quality_check','data_imputation', 'data_smoothing']

elif preprocessing_case == "processing_2":
    processing_task_list = ['data_refinement', 'data_outlier', 'data_split', 'data_integration','data_quality_check','data_imputation']

elif preprocessing_case == "processing_3":
    processing_task_list = ['data_refinement', 'data_outlier', 'data_split', 'data_integration','data_imputation']

elif preprocessing_case == "processing_4":
    processing_task_list = ['data_refinement', 'data_integration', 'data_imputation']

elif preprocessing_case == "test_data_processing_1":
    processing_task_list = ['data_refinement']
"""