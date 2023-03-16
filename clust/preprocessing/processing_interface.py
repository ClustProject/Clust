import sys
import os
sys.path.append("../")
sys.path.append("../..")
import os
import sys
sys.path.append("../")
sys.path.append("../../")

from Clust.clust.preprocessing.dataPreprocessing import DataPreprocessing
import pandas as pd

    
def get_data_result(processing_type, data_input, processing_param=None):
    """ 
    # Description       
     

    # Args
     * processing_type(_str_)    = ['refine'|'error_to_NaN'|'imputation'|'all'|'all_step_result']
     * processing_param)(dict)     
     * data_input(pandas.dataFrame or dict)

    # Returns         
     * result (pandas.dataFrame or dict)
            
    """
    
    if isinstance(data_input, dict):
        result = get_preprocessed_dataset(processing_type, processing_param, data_input)
    elif isinstance(data_input, pd.DataFrame):
        result = get_preprocessed_data(processing_type, processing_param, data_input)

    return result

def get_preprocessed_dataset(processing_type, param, data_set):
    result={}
    for key in list(data_set.keys()):
        result[key] = get_preprocessed_data(processing_type, param, data_set[key])

    return result

def get_preprocessed_all_step_result(data_input, param):
    """ Produces partial Processing data depending on process_param

    Args:
        param(dict): processing parm
        data_input (DataFrame): input data
        
    Returns:
        json: New Dataframe after preprocessing according to the process_param
        
    """
    DP = DataPreprocessing()
    
    refine_param = param['refine_param']
    outlier_param = param['outlier_param']
    imputation_param = param['imputation_param']
        
    ###########
    refined_data = DP.get_refinedData(data_input, refine_param)
    ###########
    datawithMoreCertainNaN, datawithMoreUnCertainNaN = DP.get_errorToNaNData(refined_data, outlier_param)
    ###########
    imputed_data = DP.get_imputedData(datawithMoreUnCertainNaN, imputation_param)
    ###########
    result ={'original':data_input, 'refined_data':refined_data, 'datawithMoreCertainNaN':datawithMoreCertainNaN,
    'datawithMoreUnCertainNaN':datawithMoreUnCertainNaN, 'imputed_data':imputed_data}
    
    return result
        
    
def get_preprocessed_data(processing_type, param, data):
    """ Produces only one clean data according to the processing_type and param

        Args:
            processing_type (string): ['refine'|'error_to_NaN'|'imputation'|'all']
            param (dict): parameter for preprocessing
            data (DataFrame): input data
            
            
        Returns:
            DataFrame: New Dataframe after preprocessing 
    
    """

    DP = DataPreprocessing()
    
    if processing_type =='refine':
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
    elif processing_type =='all':
        result = get_preprocessed_all_step_result(data, param)['imputed_data']
        # all_preprocessing_finalResult
        # multiDataset_all_preprocessing
    elif processing_type =='all_step_result':
        result = get_preprocessed_all_step_result(data, param)
        # all_preprocessing
        
    return result
