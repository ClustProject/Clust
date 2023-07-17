import sys
sys.path.append("../../../")
sys.path.append("../..")

import pandas as pd
from Clust.clust.meta.metaDataManager import bucketMeta
from Clust.setting import influx_setting_KETI as ins
from Clust.clust.ingestion.mongo import mongo_client
mongo_client_ = mongo_client.MongoClient(ins.CLUSTMetaInfo2)

# TODO change value
pipe_pre_case = 0

# TODO define dynamic parameter, import proper resource base on task_name
task_name = "air_quality"
level = 0

# TODO define cycle condition
cycle_condition = "week_1" # TODO change value ['week_1", "day_1"]

## TODO change case_num and cluster_num for test
case_num = 0  # change pipeline case num (0~4)
uncertain_flag = False
cluster_num = 4 # change cluster num (2~8)

# [[pipeline_case_num, clustering_flag]] -> [pipeline1-1, pipeline1-2, pipeline1-3, pipeline1-4]
case_list =[["processing_1", True], ["processing_1", False], ["processing_2", False], ["processing_3", False]]

########################################################################
#### automatically make additional variables

if task_name =='air_quality':
    from . import param_data_airquality as param_data
else:
    pass

bucket, data_param, processing_freq, feature_name  = param_data.get_data_conidtion_by_level(level)
test_pipe_param = param_data.get_data_preprocessing_param(pipe_pre_case)

if cycle_condition == "day_1":
    feature_cycle = 'Day'
    feature_cycle_times = 1 
elif cycle_condition == "week_1":
    feature_cycle = 'Week'
    feature_cycle_times = 1 
    
test_pipe_param['data_split']['split_param']['feature_cycle'] = feature_cycle
test_pipe_param['data_split']['split_param']['feature_cycle_times'] = feature_cycle_times

ingestion_method = "all_ms_in_one_bucket"
min_max = bucketMeta.get_min_max_info_from_bucketMeta(mongo_client_, bucket )

########################################################################
#### preprocessing and clustering method Setting
preprocessing_case = case_list[case_num][0]
clustering_case = case_list[case_num][1]

## 1. preprocessing param setup
if preprocessing_case == "processing_1":
    processing_task_list = ['data_refinement', 'data_outlier', 'data_split', 'data_integration','data_quality_check','data_imputation', 'data_smoothing']

elif preprocessing_case == "processing_2":
    processing_task_list = ['data_refinement', 'data_outlier', 'data_split', 'data_integration','data_imputation']

elif preprocessing_case == "processing_3":
    processing_task_list = ['data_refinement', 'data_outlier', 'data_imputation']
    
## 2. preprocessing 
processing_case_param = {
    "processing_task_list" : processing_task_list, 
    "processing_freq" : processing_freq,
    "feature_name" : feature_name,
    "data_min_max" : min_max
}

## 3. save result 
# TODO 우선 이렇게 해놨는데, 나중 테스트 확장과 결과 관리를 위해 수정할 필요가..

new_start_time = pd.to_datetime("2015-01-01 00:00:00")
new_bk_name ="task_" +task_name
cluster_result_name = new_bk_name + "_case_"+str(case_num) +"_level_"+str(level) # TODO clustering 유무 차이 넣어야함 : 함수화 할지 말지 고민 중
new_ms_name_train = cluster_result_name +'train'