import sys
sys.path.append("../")
sys.path.append("../..")
from Clust.clust.preprocessing.dataPreprocessing import DataPreprocessing
import pandas as pd

    
def get_data_result(processing_type, data_input, processing_param=None):
    """ 
    It is processing interface.
    Produce processing data according to processing_type and processing_param.

    Args
        processing_type (string) = ['refinement'|'error_to_NaN'|'certain_error_to_NaN|'uncertain_error_to_NaN'|imputation'|'step_3'|'scaling'|'smoothing']
        processing_param (dict)  or None   
        data_input (pandas.dataFrame or dict)

    Returns         
        pandas.dataFrame or dict: New Data (Dataset) after processing
            
    """
    if processing_param is None:
        processing_param = get_default_processing_param()
        
    if isinstance(data_input, dict):
        result = get_preprocessed_dataset(processing_type, processing_param, data_input)
        
    elif isinstance(data_input, pd.DataFrame):
        result = get_preprocessed_data(processing_type, processing_param, data_input)


    return result

def get_default_processing_param(min_max={'min_num':{}, "max_num":{}}, timedelta_frequency_sec=None):
    """
    get default_processing_param  (refining, min_max check(if min_max is not empty), sampling (if timedelta_frequency_sec is not None))
    
    Args:
        min_max(dict): min max restriction information of data
        timedelta_frequency_sec: timedelta for refined data processing
    Returns:
        Dictionary: default_process_param
    """
    refine_param = {"remove_duplication": {'flag': True}, "static_frequency": {'flag': True, 'frequency': timedelta_frequency_sec}}
    CertainParam= {'flag': True, 'data_min_max_limit':min_max}
    
    outlier_param ={
        "certain_error_to_NaN":CertainParam, 
        "uncertain_error_to_NaN":{'flag': False}
    }
    imputation_param = {"flag":False}
    process_param = {'refine_param':refine_param, 'outlier_param':outlier_param, 'imputation_param':imputation_param}

    return process_param
        
def get_preprocessed_dataset(processing_type, param, data_set):
    """
    Produces clean dataset consisting of multiple data according to the processing_type and param

    Args:
        processing_type (string): ['refinement'|'error_to_NaN'|'certain_error_to_NaN'|'uncertain_error_to_NaN'|'imputation'|'step_3'|'smoothing'|'scaling']
        param (dict): parameter for preprocessing
        data_set (dict): input dataset

    Returns:
        Dictionary: New Dataset after preprocessing
    
    """
    result={}
    for key in list(data_set.keys()):
        result[key] = get_preprocessed_data(processing_type, param, data_set[key])

    return result


def get_preprocessed_data(processing_type, param, data):
    """ 
    Produces only one clean data according to the processing_type and param

    Args:
        processing_type (string): ['refinement'|'error_to_NaN'|'certain_error_to_NaN'|'uncertain_error_to_NaN'|'imputation'|'step_3'|'smoothing'|'scaling']
        param (dict): parameter for preprocessing
        data (DataFrame): input data
        
    Returns:
        DataFrame: New Dataframe after preprocessing 
    
    """

    DP = DataPreprocessing()
    
    # 3 step processing
    if processing_type =='refinement':
        result = DP.get_refinedData(data, param)
    elif processing_type =='error_to_NaN':
        result_0, result = DP.get_errorToNaNData(data, param)
    elif processing_type =='certain_error_to_NaN':
        from Clust.clust.preprocessing.errorDetection.errorToNaN import errorToNaN 
        result = errorToNaN().getDataWithCertainNaN(data, param)
    elif processing_type =='uncertain_error_to_NaN':
        from Clust.clust.preprocessing.errorDetection.errorToNaN import errorToNaN 
        result = errorToNaN().getDataWithUncertainNaN(data, param)
    elif processing_type =='imputation': 
        result = DP.get_imputedData(data, param)
    elif processing_type =='step_3': # refine, error_to_NaN, certain_error_to_Nan, uncertain_error_to_Nan, impuatation
        DP = DataPreprocessing() 
        refine_param = param['refine_param']
        outlier_param = param['outlier_param']
        imputation_param = param['imputation_param']
        ###########
        refined_data = DP.get_refinedData(data, refine_param)
        ###########
        datawithMoreCertainNaN, datawithMoreUnCertainNaN = DP.get_errorToNaNData(refined_data, outlier_param)
        ###########
        result = DP.get_imputedData(datawithMoreUnCertainNaN, imputation_param)
    
    # Additional Processing    
    if processing_type =='smoothing':
        result = DP.get_smoothed_data(data, param)
    if processing_type =='scaling':
        result = DP.get_scaling_data(data, param)
    
    return result
