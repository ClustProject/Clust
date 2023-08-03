import sys
sys.path.append("../../../")
sys.path.append("../..")

import pandas as pd
from dateutil.relativedelta import relativedelta

from Clust.clust.meta.metaDataManager import bucketMeta
from Clust.setting import influx_setting_KETI as ins
from Clust.clust.ingestion.mongo import mongo_client
mongo_client_ = mongo_client.MongoClient(ins.CLUSTMetaInfo2)

# TODO eda flag
eda_flag = True

# TODO change value
pipe_pre_case = 3

# TODO define dynamic parameter, import proper resource base on task_name
task_name = "air_quality"
data_level = 13

# TODO define cycle condition
cycle_condition = "week_1" # TODO change value ['week_1", "day_1"]

## TODO change case_num and cluster_num for test
case_num = 1  # change pipeline case num (0~4)
uncertain_flag = False
cluster_num = 8 # change cluster num (2~8)

# [[pipeline_case_num, clustering_flag]] -> [pipeline1-1, pipeline1-2, pipeline1-3, pipeline1-4, test pipeline]
case_list =[["processing_1", True], ["processing_2", False], ["processing_3", False], ["processing_4", False], ["test_data_processing_1", False]]

########################################################################
#### automatically make additional variables

if task_name =='air_quality':
    from . import param_data_airquality as param_data
else:
    pass

bucket, data_param, processing_freq, feature_name, ingestion_method  = param_data.get_data_conidtion_by_data_level(data_level)
test_pipe_param = param_data.get_data_preprocessing_param(pipe_pre_case)

if cycle_condition == "day_1":
    feature_cycle = 'Day'
    feature_cycle_times = 1 
elif cycle_condition == "week_1":
    feature_cycle = 'Week'
    feature_cycle_times = 1 
    
test_pipe_param['data_split']['split_param']['feature_cycle'] = feature_cycle
test_pipe_param['data_split']['split_param']['feature_cycle_times'] = feature_cycle_times

min_max = bucketMeta.get_min_max_info_from_bucketMeta(mongo_client_, bucket )

########################################################################
#### preprocessing and clustering method Setting
preprocessing_case = case_list[case_num][0]
clustering_case = case_list[case_num][1]

## 1. preprocessing param setup
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
## 2. preprocessing 
processing_case_param = {
    "processing_task_list" : processing_task_list, 
    "processing_freq" : processing_freq,
    "feature_name" : feature_name,
    "data_min_max" : min_max
}

## 3. save result 
delta = relativedelta(years=5)
new_start_time = data_param['start_time'] - delta
new_bk_name ="task_" + task_name
cluster_result_name = new_bk_name + "_case_"+str(case_num) +"_level_"+str(data_level)+"_pre_param_"+str(pipe_pre_case)+"_clustering_"+str(clustering_case) # TODO 

def get_new_ms_name(data_type, select_class = None, cluster_result_name = cluster_result_name):
    if select_class:
        select_class_str = ''.join(str(c) for c in select_class)
        cluster_result_name = cluster_result_name +"_"+select_class_str
    new_ms_name = cluster_result_name + "_" + data_type

    return new_ms_name