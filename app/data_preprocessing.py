import pandas as pd
import sys
sys.path.append("../")


def get_process_param_by_level(level):
    """
    Args:
        level (int): cleaning level
    Returns:
        process_param(dict): process_param 
    """
    refine_param = {"removeDuplication": {"flag": False},"staticFrequency": {"flag": False, "frequency": None}}
    certain_param = {'flag': False}
    uncertain_param = {'flag': False}
    imputation_param = {"flag": False}
    
    if level == 0:
        pass 
    if level >= 1:
        refine_param = {"removeDuplication": {"flag": True},"staticFrequency": {"flag": True, "frequency": None}}
        
    if level >= 2:
        uncertain_param['flag'] = True
        
    if level >= 3:
        imputation_param = {
            "flag": False,
            "imputation_method": [{"min": 0, "max": 2, "method": "linear", "parameter": {}}],
            "totalNonNanRatio": 90
        }
        
    if level >= 4:
        uncertain_param = {'flag': True, "param": {
            "outlierDetectorConfig": [{'algorithm': 'IQR', 'percentile': 99,'alg_parameter': {'weight': 100}}]}}

    process_param = {'refine_param': refine_param,
                     'outlier_param': {"certainErrorToNaN":  certain_param, "unCertainErrorToNaN": uncertain_param}, 
                     'imputation_param': imputation_param}

    return process_param
    