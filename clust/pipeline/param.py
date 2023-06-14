import sys
sys.path.append("../")
sys.path.append("../..")
import datetime


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
    default_param['data_split']={
        "split_method":"cycle",
        "split_param":{
            'feature_cycle' : "Day",
            'feature_cycle_times' : 1}
    }
    
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


