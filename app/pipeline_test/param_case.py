import sys
sys.path.append("../../../")
sys.path.append("../..")

import pandas as pd
from dateutil.relativedelta import relativedelta



########################################################################
#### automatically make additional variables
task_name = "air_quality"

if task_name =='air_quality':
    from . import param_data_airquality as param_data
else:
    pass


def get_processing_task_list(preprocessing_case, data_param, test_pipe_param):
    if preprocessing_case == "processing_1":
        processing_task_list = ['data_outlier', 'data_refinement', 'data_split', 'data_integration','data_quality_check','data_imputation', 'data_smoothing'] # +clustering

    elif preprocessing_case == "processing_2":
        processing_task_list = [ 'data_outlier', 'data_refinement','data_split', 'data_integration','data_quality_check','data_imputation']

    elif preprocessing_case == "processing_3":
        processing_task_list = ['data_refinement', 'data_outlier', 'data_imputation']

    elif preprocessing_case == "processing_4":
        test_pipe_param['data_integration']['integration_param']['duration'] = {'start_time': data_param['start_time'], 'end_time': data_param['end_time']}
        processing_task_list = ['data_refinement',  'data_imputation']

    return processing_task_list, test_pipe_param


def get_test_pipe_param(consecutive_nan_limit_number, cycle_condition):
    test_pipe_param = param_data.get_data_preprocessing_param(consecutive_nan_limit_number)

    if cycle_condition == "day_1":
        feature_cycle = 'Day'
        feature_cycle_times = 1 
        
    elif cycle_condition == "week_1":
        feature_cycle = 'Week'
        feature_cycle_times = 1 
        
    test_pipe_param['data_split']['split_param']['feature_cycle'] = feature_cycle
    test_pipe_param['data_split']['split_param']['feature_cycle_times'] = feature_cycle_times
    
    return test_pipe_param

def get_new_ms_name(cluster_result_name, select_class = None):
    if select_class:
        select_class_str = ''.join(str(c) for c in select_class)
        cluster_result_name = cluster_result_name +"_"+select_class_str
    new_ms_name = cluster_result_name 

    return new_ms_name