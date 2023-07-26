import sys
sys.path.append("../")
sys.path.append("../..")
sys.path.append("../../..")
from Clust.clust.pipeline import param
import math
import matplotlib.pyplot as plt

def pipeline_eda(data, module_list, feature_name):
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
            ...     ['data_split', split_param],
            ...     ['data_selection', select_param],
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
            plot_dataset(data, module_name, feature_name)

        if module_name =="data_outlier":
            module_param = param.set_outlier_param(module_param)
            from Clust.clust.preprocessing import processing_interface
            data= processing_interface.get_data_result('error_to_NaN', data, module_param)
            plot_dataset(data, module_name, feature_name)
            
        elif module_name =='data_split':          
            split_method = module_param['split_method']
            split_param = module_param['split_param']
            from Clust.clust.transformation.general import split_interface
            data = split_interface.get_data_result(split_method, data, split_param)
            plot_dataset(data, module_name, feature_name)
            
        elif module_name == 'data_selection':          
            from Clust.clust.transformation.general import select_interface
            select_method = module_param['select_method']
            select_param = module_param['select_param']
            data = select_interface.get_data_result(select_method, data,  select_param)
            plot_dataset(data, module_name, feature_name)
            
        elif module_name =='data_integration':
            integration_type =module_param['integration_type']
            integration_param = module_param['integration_param']
            from Clust.clust.integration import integration_interface
            data = integration_interface.get_data_result(integration_type, data, integration_param)
            print(data.isna().sum())
            
        elif module_name == 'data_quality_check':
            from Clust.clust.quality import quality_interface
            quality_method = module_param['quality_method']
            quality_param = module_param['quality_param']
            data = quality_interface.get_data_result(quality_method, data, quality_param)
            print(data.isna().sum())
            
        elif module_name == 'data_imputation': 
            from Clust.clust.preprocessing import processing_interface
            data = processing_interface.get_data_result('imputation', data, module_param)
            plot_data_by_column(data, module_name)
            print(data.isna().sum())
            
        elif module_name =='data_smoothing':
            from Clust.clust.preprocessing import processing_interface
            data = processing_interface.get_data_result('smoothing', data, module_param)
            plot_data_by_column(data, module_name)
            
        elif module_name =='data_scaling': 
            from Clust.clust.preprocessing import processing_interface
            data = processing_interface.get_data_result('scaling', data, module_param)
        
        print(get_shape(data))
        if isinstance(data, dict):
            for processing_data in data.values():
                if processing_data.empty:
                    print("========= pipeline stop ::: data is empty =========")
                    break

        elif isinstance(data, pd.DataFrame):
            if data.empty:
                print("========= pipeline stop ::: data is empty =========")
                break
        
    return data


import pandas as pd
def get_shape(data):
    """data의 형태를 리턴함

    Args:
        data (dataFrame or dataFrameSet): 입력 데이터

    Returns:
        data shape : 각 데이터의 쉐입을 전달함
    """
    if isinstance(data, pd.DataFrame):
        return data.shape
    else:
        d1 = len(data)
        first_key = list(data.keys())[0]
        d2 = data[first_key].shape
        return (d1, d2)

def plot_dataset(dataset, module_name, feature_name=None):
    row = math.ceil(len(dataset)/2)
    plt.figure(figsize=(20,row*3.5))
    for i, data_name in enumerate(dataset):
        data = dataset[data_name]
        plt.rc('font', size=8)
        plt.subplot(row, 2, i+1)
        plt.title("{}_{}".format(data_name, module_name))
        plt.plot(data)
        if feature_name:
            print("{}_{}_Nan : ".format(data_name, feature_name), data.isna().sum()[feature_name])

def plot_data_by_column(data, module_name):
    row = math.ceil(len(data.columns)/2)
    plt.figure(figsize=(20,row*3))
    for i, data_name in enumerate(data):
        column_data = data[data_name]
        plt.rc('font', size=8)
        plt.subplot(row, 2, i+1)
        plt.title("{}_{}".format(data_name, module_name))
        plt.plot(column_data)