import pandas as pd
# Good: 2-3, 6-7, 8-9, 16-17(Best)
# Bad: 4-5, 22- 23
# (test) 0-1
def get_data_conidtion_by_data_level(data_level = 0):
    ##############################################################################################
    data_param={}
    
    if data_level in [0, 1]:
        feature_name = 'in_noise' # integration, prediction feature
        processing_freq = 1 # refinement, integration frequency
        bucket = 'air_indoor_중학교'
        data_param['bucket_name'] = bucket
        data_param['feature_list']= [feature_name]
        
        if data_level == 0: #train
            ingestion_method = "all_ms_in_one_bucket"
            data_param['start_time'] = pd.to_datetime("2022-05-01 00:00:00")
            data_param['end_time'] = pd.to_datetime("2022-08-30 23:59:59")

        elif data_level == 1: #test
            ingestion_method = "all_ms_in_one_bucket"
            data_param['start_time'] = pd.to_datetime("2022-09-01 00:00:00")
            data_param['end_time'] = pd.to_datetime("2022-09-14 23:59:59")
    
    if data_level in [2, 3]:
        processing_freq = 10 # refinement, integration frequency
        feature_name = 'in_co2'
        bucket = 'air_indoor_체육시설'  
        data_param['bucket_name'] = bucket
        
        if data_level == 2:
            ingestion_method = "all_ms_in_one_bucket"
            data_param['start_time']= pd.to_datetime("2021-01-01 00:00:00")
            data_param['end_time'] = pd.to_datetime("2021-08-31 23:59:59")
        elif data_level == 3:
            ingestion_method = "multiple_ms_by_time"
            data_param['ms_list_info'] = [[bucket, 'ICW0W2001037'], [bucket, 'ICW0W2001044']]
            data_param['feature_list']= [[feature_name], [feature_name]]
            data_param['start_time'] = pd.to_datetime("2021-09-01 00:00:00")
            data_param['end_time'] = pd.to_datetime("2021-09-20 23:59:59")
            
    if data_level in [4, 5]:
        bucket ='air_indoor_초등학교'
        data_param['bucket_name'] = bucket
        feature_name = 'in_co2' # integration, prediction feature
        processing_freq = 10 # refinement, integration frequency
        
        if data_level == 4:
            ingestion_method = "all_ms_in_one_bucket"
            data_param['start_time']= pd.to_datetime("2021-03-01 00:00:00")
            data_param['end_time'] = pd.to_datetime("2021-10-31 23:59:59")
            
        elif data_level == 5:
            ingestion_method = "multiple_ms_by_time"
            data_param['start_time']= pd.to_datetime("2021-11-01 00:00:00")
            data_param['end_time'] = pd.to_datetime("2021-11-26 23:59:59")
            data_param['ms_list_info'] = [[bucket, 'ICW0W2000023'], [bucket, 'ICW0W2000024'], [bucket, 'ICW0W2000025'], [bucket, 'ICW0W2000031'], [bucket, 'ICW0W2000034']]
            data_param['feature_list']= [[feature_name], [feature_name], [feature_name], [feature_name], [feature_name]]
      
    if data_level in [6, 7]:
        bucket =  'air_indoor_초등학교'
        data_param['bucket_name'] = bucket
        feature_name = 'in_co2' # integration, prediction feature
        processing_freq = 1 # refinement, integration frequency
        if data_level == 6:
            data_param['start_time']= pd.to_datetime("2021-07-01 00:00:00")
            data_param['end_time'] = pd.to_datetime("2021-10-31 23:59:59")
            ingestion_method = "all_ms_in_one_bucket"
        
        elif data_level == 7:
            data_param['start_time']= pd.to_datetime("2021-11-01 00:00:00")
            data_param['end_time'] = pd.to_datetime("2021-11-26 23:59:59")
            
            data_param['ms_list_info'] = [[bucket, 'ICW0W2000023'], [bucket, 'ICW0W2000024'], [bucket, 'ICW0W2000025'], [bucket, 'ICW0W2000031'], [bucket, 'ICW0W2000034']]
            data_param['feature_list']= [[feature_name], [feature_name], [feature_name], [feature_name], [feature_name]]
            ingestion_method = "multiple_ms_by_time"
    
    if data_level in [8, 9]:
        bucket = 'air_indoor_경로당'
        data_param['bucket_name'] = bucket # 경로당 좋지 않은 데이터가 많음
        processing_freq = 10
        feature_name = 'in_co2'
        if data_level == 9:
            ingestion_method = "multiple_ms_by_time"
            data_param['start_time']= pd.to_datetime("2021-10-01 00:00:00")
            data_param['end_time'] = pd.to_datetime("2021-10-31 23:59:59")
            data_param['ms_list_info'] = [[bucket, 'ICL1L2000251'], [bucket, 'ICL1L2000271'], [bucket, 'ICL1L2000238'], [bucket, 'ICL1L2000279'], 
                                        [bucket, 'ICL1L2000276'], [bucket, 'ICL1L2000240'], [bucket, 'ICL1L2000242'], [bucket, 'ICL1L2000252']]
            
            data_param['feature_list']= [[feature_name], [feature_name], [feature_name], [feature_name], [feature_name], [feature_name], [feature_name], [feature_name]]           

        elif data_level == 10:
            ingestion_method = "ms_by_time"
            data_param['start_time']= pd.to_datetime("2021-11-08 00:00:00")
            data_param['end_time'] = pd.to_datetime("2021-11-14 23:59:59")
            data_param['ms_name'] = "ICL1L2000252"
            data_param['feature_list'] = [feature_name]      
            
    if data_level in [16, 17]:
        # 16, 17 유사 버전에 대해서 (116, 117 216 217등으로 복제 및 수정하여) 해보셔요 , frequency, feature등 바꿔가며
        bucket = 'air_indoor_경로당'
        data_param['bucket_name'] = bucket
        processing_freq = 1
        feature_name = 'in_co2'
        data_param['feature_list']= [feature_name]
        if data_level == 16:
            data_param['start_time']= pd.to_datetime("2021-10-01 00:00:00")
            data_param['end_time'] = pd.to_datetime("2021-10-31 23:59:59")
            ingestion_method = "all_ms_in_one_bucket"
            """
            ingestion_method = "multiple_ms_by_time"
            data_param['ms_list_info'] = [[bucket, 'ICL1L2000251'], [bucket, 'ICL1L2000252'], [bucket, 'ICL1L2000275'], [bucket, 'ICL1L2000277'], [bucket, 'ICL1L2000279']]
            data_param['feature_list']= [[feature_name], [feature_name], [feature_name], [feature_name], [feature_name]]
            """
        elif data_level == 17:
            data_param['start_time']= pd.to_datetime("2021-11-01 00:00:00")
            data_param['end_time'] = pd.to_datetime("2021-11-12 23:59:59")
            ingestion_method = "all_ms_in_one_bucket"
            """
            data_param['start_time']= pd.to_datetime("2021-11-08 00:00:00")
            data_param['end_time'] = pd.to_datetime("2021-11-2 23:59:59")
            ingestion_method = "ms_by_time"
            data_param['ms_name'] = "ICL1L2000252"
            data_param['feature_list'] = [feature_name]
            """
    if data_level in [53, 54]:
        feature_name = 'in_co2' # integration, prediction feature
        processing_freq = 5 # refinement, integration frequency\
        bucket ='air_indoor_체육시설'
        data_param['bucket_name'] = bucket
        if data_level == 53:
            data_param['start_time']= pd.to_datetime("2021-01-01 00:00:00")
            data_param['end_time'] = pd.to_datetime("2021-08-31 23:59:59")
            ingestion_method = "all_ms_in_one_bucket"

        elif data_level == 54:
            data_param['start_time'] = pd.to_datetime("2021-09-01 00:00:00")
            data_param['end_time'] = pd.to_datetime("2021-09-20 23:59:59")
            ingestion_method = "multiple_ms_by_time"
            data_param['ms_list_info'] = [[bucket, 'ICW0W2001037'], [bucket, 'ICW0W2001044']]
            data_param['feature_list']= [[feature_name], [feature_name]]
    ## JW
    if data_level in [20, 21]:
        bucket = 'air_indoor_경로당'
        data_param['bucket_name'] = bucket
        feature_name = 'in_co2'
        processing_freq = 10
        if data_level == 20:
            data_param['start_time']= pd.to_datetime("2021-09-15 00:00:00")
            data_param['end_time'] = pd.to_datetime("2021-10-15 23:59:59")
            data_param['feature_list'] = [feature_name]
            ingestion_method = "all_ms_in_one_bucket"
        elif data_level == 21:
            data_param['start_time']= pd.to_datetime("2021-10-16 00:00:00")
            data_param['end_time'] = pd.to_datetime("2021-10-30 23:59:59")
            ingestion_method = "all_ms_in_one_bucket"
    
    if data_level in [22, 23]:
        bucket = 'air_indoor_경로당'
        data_param['bucket_name'] = bucket
        processing_freq = 60
        feature_name = 'in_temp'
        if data_level == 22:
            data_param['start_time']= pd.to_datetime("2021-09-15 00:00:00")
            data_param['end_time'] = pd.to_datetime("2021-10-15 23:59:59")
            ingestion_method = "all_ms_in_one_bucket"
            data_param['feature_list'] = [feature_name]
            
        elif data_level == 23:
            data_param['start_time']= pd.to_datetime("2021-10-16 00:00:00")
            data_param['end_time'] = pd.to_datetime("2021-11-15 23:59:59")
            ingestion_method = "multiple_ms_by_time"
            data_param['ms_list_info'] = [[bucket, 'ICL1L2000271']]
            data_param['feature_list']= [[feature_name], [feature_name], [feature_name], [feature_name], [feature_name], [feature_name], [feature_name], [feature_name], [feature_name]]      
        
    return bucket, data_param, processing_freq, feature_name, ingestion_method


def get_data_preprocessing_param(consecutive_nan_limit_number):
    quality_limit = consecutive_nan_limit_number
    total_nan_limit = 100000
    total_imputation_limit = 100000000000
    pipe_param = {
        'data_refinement': { 
                "remove_duplication": {'flag': True},
                "static_frequency": {'flag': True, 'frequency': None}
        },
        'data_split': {
            'split_method': 'cycle',
            'split_param': {'feature_cycle': None, 'feature_cycle_times': None}
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
            'quality_param': {'nan_processing_param': {'type': 'num','ConsecutiveNanLimit': quality_limit,'totalNaNLimit': total_nan_limit}}
        },
        'data_imputation': {'flag': True,
                                'imputation_method': [{'min': 0,'max': total_imputation_limit,'method': 'linear','parameter': {}}],
                                'total_non_NaN_ratio': 1},
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
        'data_smoothing': {"flag": True, "emw_param":0.3},
        'data_scaling': {'flag': True, 'method':'minmax'} 
    }
    
    return pipe_param
        

