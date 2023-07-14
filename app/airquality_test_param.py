# TODO 지수 제가 시간이 없어서 정리를 하다 말았어요. 이 파일만 고치면 수월하게 되도록 부탁
# 각 Test에 대해 체계적 테스트 되도록

import sys
sys.path.append("../../../")
sys.path.append("../..")

import pandas as pd
from Clust.setting import influx_setting_KETI as ins
from Clust.clust.ingestion.mongo import mongo_client
mongo_client_ = mongo_client.MongoClient(ins.CLUSTMetaInfo2)
from Clust.clust.meta.metaDataManager import bucketMeta

########################################################################################################################################################################
########################### data ingestion parameter setting
data_param={}
bucket ='air_indoor_체육시설'
ingestion_method = "all_ms_in_one_bucket"
data_param['start_time']= pd.to_datetime("2021-08-01 00:00:00")
data_param['end_time'] = pd.to_datetime("2021-08-31 23:59:59")
data_param['bucket_name'] = bucket
###########################

###########################  specification test parameter condition and number setting
test_pipe_param_num = 1
if test_pipe_param_num == 1:
    feature_cycle = 'Day'
    feature_cycle_times = 1 

if test_pipe_param_num == 2:
    feature_cycle = 'Week'
    feature_cycle_times = 1 

############################ preprocessing and clustering method Setting
case_list =[["processing_1", True], ["processing_1", False], ["processing_2", False], ["processing_3", False]]
case_num = 0

preprocessing_case = case_list[case_num][0]
clustering_case = case_list[case_num][1]
cluster_num = 4

cluster_result_name = "test_"+str(case_num) # 데이터, 파일 이름등이 더 많은 정보를 담고 차별될 수 있도록 수정해야함 우선 뒀음

if preprocessing_case == "processing_1":
    processing_task_list = ['data_refinement', 'data_outlier', 'data_split', 'data_integration','data_quality_check','data_imputation', 'data_smoothing']

elif preprocessing_case == "processing_2":
    processing_task_list = ['data_refinement', 'data_outlier', 'data_split', 'data_integration','data_imputation']

elif preprocessing_case == "processing_3":
    processing_task_list = ['data_refinement', 'data_outlier', 'data_imputation']

############################ preprocessing parameter setting (common)
processing_freq = 10 # refinement, integration frequency
feature_name = 'in_co2' # integration, prediction feature
############################

############################ uncertain parameter setting (위 파라미터에 통합하는게 좋을 듯) - outlier detection
uncertain_flag = False
min_max = bucketMeta.get_min_max_info_from_bucketMeta(mongo_client_, data_param['bucket_name'] )

############################ new data saving
new_start_time = pd.to_datetime("2015-01-01 00:00:00")

## bucket name 은 고정
new_data_bk_name = "kweather_data_regression_test_jw" # <<- 이런것들 체계적으로 수정
## ms name 만 수정
new_data_train_ms_name = bucket + "all_ms"+ "train_casenum_" + str(case_num)

########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

processing_case_param = {
    "processing_task_list" : processing_task_list, 
    "processing_freq" : processing_freq,
    "feature_name" : feature_name,
    "data_min_max" : min_max
}
############################
test_pipe_param = {
    'data_refinement': { 
        "remove_duplication": {'flag': True},
        "static_frequency": {'flag': True, 'frequency': None}
    }, 
    'data_outlier': {
        'certain_error_to_NaN': {'flag': True},
        'uncertain_error_to_NaN': 
        {
             'flag': True,
             'outlier_detector_config':{
                 'algorithm': 'SR', 
                 'percentile': 95,
                 'alg_parameter': {'period': 144} ## algorithm에 따라 아래 입력 방식(입력 받는 key, value)이 달라짐. 아래를 참고
             }
        }
    },
    'data_split': {
        'split_method': 'cycle',
        'split_param': {'feature_cycle': feature_cycle, 'feature_cycle_times': feature_cycle_times}
    },
    'data_selection': {'select_method': 'keyword_data_selection',
                       'select_param': {'keyword': '*'}
    },
    
    'data_integration': {
        'integration_type': 'one_feature_based_integration',
        'integration_param': {'feature_name': None,'duration': None,'integration_frequency': None }
    },
    'data_quality_check': {
        'quality_method': 'data_with_clean_feature',
        'quality_param': {'nan_processing_param': {'type': 'num','ConsecutiveNanLimit': 30,'totalNaNLimit': 181}}
    },
    'data_imputation': {'flag': True,
                        'imputation_method': [{'min': 0,'max': 300,'method': 'linear','parameter': {}}],
                        'total_non_NaN_ratio': 1},
    # - data_smoothing 선택시 
    ## 사용자 input: data_smoothing.emw_param (float: 0~1)
    'data_smoothing': {"flag": True, "emw_param":0.3},
    
    # - data_scaling 선택시 사용자 input 없음
    'data_scaling': {'flag': True, 'method':'minmax'} 
}
    