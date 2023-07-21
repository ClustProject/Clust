import pandas as pd

def get_data_conidtion_by_data_level(data_level = 0):
    ##############################################################################################
    data_param={}

    # set Level 0 
    if data_level == 0:
        bucket ='air_indoor_체육시설'
        data_param['start_time']= pd.to_datetime("2021-08-01 00:00:00")
        data_param['end_time'] = pd.to_datetime("2021-08-31 23:59:59")
        data_param['bucket_name'] = bucket
        processing_freq = 10 # refinement, integration frequency
        # feature
        feature_name = 'in_co2' # integration, prediction feature
        # ingestion method
        ingestion_method = "all_ms_in_one_bucket"
        
    elif data_level == 1:
        bucket ='air_indoor_체육시설'
        data_param['start_time']= pd.to_datetime("2021-08-01 00:00:00")
        data_param['end_time'] = pd.to_datetime("2021-08-31 23:59:59")
        data_param['bucket_name'] = bucket
        processing_freq = 10 # refinement, integration frequency
        # feature
        feature_name = 'in_noise' # integration, prediction feature
        # ingestion method
        ingestion_method = "all_ms_in_one_bucket"

    elif data_level == 2:
        # test data ingestion info : ingestion_method(multiple_ms_by_time)
        bucket ='air_indoor_체육시설'
        data_param['start_time'] = pd.to_datetime("2021-09-27 00:00:00")
        data_param['end_time'] = pd.to_datetime("2021-10-22 23:59:59")
        data_param['ms_list_info'] = [[bucket, 'ICW0W2001037'], [bucket, 'ICW0W2001044']]
        processing_freq = 10 # train data 와 freq를 맞춰주기 위해서 필요
        # feature
        feature_name = 'in_co2'
        data_param['feature_list']= [[feature_name], [feature_name]]
        # ingestion method
        ingestion_method = "multiple_ms_by_time"

    elif data_level == 3:
        bucket ='air_indoor_체육시설'
        data_param['start_time']= pd.to_datetime("2021-01-01 00:00:00")
        data_param['end_time'] = pd.to_datetime("2021-08-31 23:59:59")
        data_param['bucket_name'] = bucket
        processing_freq = 10 # refinement, integration frequency
        # feature
        feature_name = 'in_co2' # integration, prediction feature
        # ingestion method
        ingestion_method = "all_ms_in_one_bucket"

    elif data_level == 4:
        bucket ='air_indoor_초등학교'
        data_param['start_time']= pd.to_datetime("2021-03-01 00:00:00")
        data_param['end_time'] = pd.to_datetime("2021-10-31 23:59:59")
        data_param['bucket_name'] = bucket
        processing_freq = 10 # refinement, integration frequency
        # feature
        feature_name = 'in_co2' # integration, prediction feature
        # ingestion method
        ingestion_method = "all_ms_in_one_bucket"
    
    elif data_level == 5:
        bucket ='air_indoor_초등학교'
        data_param['start_time']= pd.to_datetime("2021-11-01 00:00:00")
        data_param['end_time'] = pd.to_datetime("2021-11-26 23:59:59")
        data_param['ms_list_info'] = [[bucket, 'ICW0W2000023'], [bucket, 'ICW0W2000024'], [bucket, 'ICW0W2000025'], [bucket, 'ICW0W2000031'], [bucket, 'ICW0W2000034']]
        #data_param['bucket_name'] = bucket
        processing_freq = 10 # refinement, integration frequency
        # feature
        feature_name = 'in_co2' # integration, prediction feature
        data_param['feature_list']= [[feature_name], [feature_name], [feature_name], [feature_name], [feature_name]]
        # ingestion method
        ingestion_method = "multiple_ms_by_time"
        
    return bucket, data_param, processing_freq, feature_name, ingestion_method


def get_data_preprocessing_param(case):
    if case == 0:
        pipe_param = {
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
        
        
    return pipe_param
        

