import sys
sys.path.append("../")
sys.path.append("../..")
import datetime


def pipeline(data, module_list):
    """ 
    data pipeline 작업을 수행함 (다수개의 모듈연산(모듈+모듈의 파라미터)을 순서대로 수행)

    Args:
        data(dataframe or dictionary (set of dataframe)) : input data
        pipleline_list(list of list) : list of [module_name, module_param] 
    
    Returns:
        DataFrame or Dictionary : data

    Example:
            >>> pipeline_list = 
            ...     [['data_refinement', refine_param],
            ...     ['data_outlier', outlier_param],
            ...     ['data_split', holiday_split_param],
            ...     ['data_selection', select_param],
            ...     ['data_split', cycle_split_param],
            ...     ['data_integration', integration_param],
            ...     ['data_quality_check', quality_param],
            ...     ['data_imputation', imputation_param],
            ...     ['data_smoothing', smoothing_param],
            ...     ['data_scaling', scaling_param]]

      
    """
    for module in module_list:
        module_name, module_param = module[0], module[1]
        
        print(module_name) 
        
        if module_name == 'data_refinement': 
            from Clust.clust.preprocessing import processing_interface
            data= processing_interface.get_data_result('refinement', data, module_param)
            
        if module_name =="data_outlier":
            from Clust.clust.preprocessing import processing_interface
            data= processing_interface.get_data_result('error_to_NaN', data, module_param)
            
        elif module_name =='data_split':           
            split_method = module_param['split_method']
            split_param = module_param['split_param']
            from Clust.clust.transformation.general import split_interface
            data = split_interface.get_data_result(split_method, data, split_param)
            
        elif module_name == 'data_selection':          
            from Clust.clust.transformation.general import select_interface
            select_method = module_param['select_method']
            select_param = module_param['select_param']
            data = select_interface.get_data_result(select_method, data,  select_param)
            
        elif module_name =='data_integration':
            integration_type =module_param['integration_type']
            integration_param = module_param['integration_param']
            from Clust.clust.integration import integration_interface
            data = integration_interface.get_data_result(integration_type, data, integration_param)
            
        elif module_name == 'data_quality_check':
            from Clust.clust.quality import quality_interface
            quality_method = module_param['quality_method']
            quality_param = module_param['quality_param']
            data = quality_interface.get_data_result(quality_method, data, quality_param)
            
        elif module_name == 'data_imputation': 
            from Clust.clust.preprocessing import processing_interface
            data = processing_interface.get_data_result('imputation', data, module_param)
            
        elif module_name =='data_smoothing':
            data = processing_interface.get_data_result('smoothing', data, module_param)
            
        elif module_name =='data_scaling': 
            data = processing_interface.get_data_result('scaling', data, module_param)

        print(get_shape(data))
        
    return data


            
def set_default_param():
    default_param={}
    ## 1. refine_param
    data_freq_min = 60
    refine_frequency = datetime.timedelta(minutes= data_freq_min)

    default_param['data_refinement'] = {"remove_duplication": {'flag': True}, 
                    "static_frequency": {'flag': True, 'frequency': refine_frequency}}
    
    ## 2. outlier_param

    default_param['data_outlier'] ={
        "certain_error_to_NaN": {'flag': True, }, 
        "uncertain_error_to_NaN":{'flag': False}}
    
    ## 3. split_param
    default_param['data_split']['cycle']={
        "split_method":"cycle",
        "split_param":{
            'feature_cycle' : "Day",
            'feature_cycle_times' : 1}
    }
    #default_param['data_split']['holiday']
    
    ## 4. select_param
    default_param['data_selection']={
        "select_method":"keyword_data_selection", 
        "select_param":{
            "keyword":"*"
        }
    }
    
    ## 5. integration_param
    data_freq_min = 60 
    integration_frequency = datetime.timedelta(minutes= data_freq_min)

    default_param['data_integration']={
        "integration_param":{"feature_name":"in_co2", "duration":None, "integration_frequency":integration_frequency},
        "integration_type": "one_feature_based_integration"
    }
    
    ## 6. quality_param
    default_param['data_quality_check'] = {
        "quality_method" : "data_with_clean_feature", 
        "quality_param" : {
            "nan_processing_param":{
                'type':"num", 
                'ConsecutiveNanLimit':4, 
                'totalNaNLimit':24}}
    }
    
    ## 7. imputation_param
    default_param['data_imputation'] = {
                        "flag":True,
                        "imputation_method":[{"min":0,"max":300,"method":"linear", "parameter":{}}, 
                                            {"min":0,"max":10000,"method":"mean", "parameter":{}}],
                        "total_non_NaN_ratio":1 }
    
    ## 8. smoothing_param
    default_param['data_smoothing']={'flag': True, 'emw_param':0.3}
    
    ## 9. scaling_param
    default_param['data_scaling']={'flag': True, 'method':'minmax'} 
    return default_param

import pandas as pd
def get_shape(data):
    if isinstance(data, pd.DataFrame):
        return data.shape
    else:
        d1 = len(data)
        first_key = list(data.keys())[0]
        d2 = data[first_key].shape
        return (d1, d2)