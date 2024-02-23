import sys
sys.path.append("../")
sys.path.append("../..")
from . import param
DF = "DF"
DFSet ="DFSet"
# Pipeline에 대해 각 모듈별 가능한 인풋과 아웃풋의 연결을 서술하는 전역 변수로 추후 수정될 수 있음
pipeline_rule = {
    "data_refinement":{
        DF:DF, 
        DFSet:DFSet
    },
    "data_outlier":{
        DF:DF, 
        DFSet:DFSet
    },
    "data_imputation":{
        DF:DF, 
        DFSet:DFSet
    },
    "data_smoothing":{
        DF:DF, 
        DFSet:DFSet
    },
    "data_scaling":{
        DF:DF,  
        DFSet:DFSet
    },
    "data_split":{
        DF:DFSet,
        DFSet:DFSet
    },
    "data_selection":{
        DFSet:DFSet
    },
    "data_integration":{
        DFSet:DF
    },
    "data_quality_check":{
        DF:DF
    },
    "data_flattening":{
        DF:DF
    }

}

def pipeline(data, module_list, edafalg =False):
    """ 
    data pipeline 작업을 수행함 (다수개의 모듈연산(모듈+모듈의 파라미터)을 순서대로 수행), feature name을 입력하는 경우 이에 대한 specific EDA 수행

    Args:
        data(dataframe or dictionary (set of dataframe)) : input data
        pipleline_list(list of list) : list of [module_name, module_param] 
        edafalg: eda flag
    
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
            ...     ['data_scaling', scaling_param],
            ...     ['data_flattening', flattening_param]]

      
    """
    print("############################################ module_name:", "Original") 
    if edafalg:
        pipeline_result_EDA(data, "Original")
        print("Original Shape: ", get_shape(data))
        
    for module in module_list:
        module_name, module_param = module[0], module[1]
        print("############################################ module_name:", module_name) 
        if module_name == 'data_refinement': 
            from Clust.clust.preprocessing import processing_interface
            data= processing_interface.get_data_result('refinement', data, module_param)

        if module_name =="data_outlier":
            module_param = param.set_outlier_param(module_param)
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
            from Clust.clust.preprocessing import processing_interface
            data = processing_interface.get_data_result('smoothing', data, module_param)

        elif module_name =='data_scaling': 
            from Clust.clust.preprocessing import processing_interface
            data = processing_interface.get_data_result('scaling', data, module_param)

        elif module_name == 'data_flattening':
            from Clust.clust.transformation.general import flatten_interface
            data = flatten_interface.make_uni_variate_with_time_index(data, module_param)
        
        if edafalg: #우선 EDA 선별자
            pipeline_result_EDA(data, module_name)
            
        print("after module processing shape:", get_shape(data))
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

def pipeline_connection_check(pipeline, input_type):
    """pipeline의 in-output module 결합 유효성을 검사한다.

    Args:
        module1 (string): 선모듈 이름
        module2 (string): 후모듈 이름
        input_type (string): 데이터 인풋타입, "DF" or "DFSet"
        
     Returns:
        Bool : output_type (True or False, 유효성 여부 전달)
    """
    for pipe in pipeline:
        method = pipe[0]
        output_type = pipeline_module_check(method, input_type)
        if output_type:
            input_type = output_type
            valid = True
        else:
            print(method, "is not working. ", "Input_type is", input_type)
            valid = False
            break
        
    return valid

def pipeline_module_check(method, input_type):
    """pipeline module의 인풋과 메소드가 유효한지를 체크함, 유효한 경우 output type을 뱉고 유효하지 않을 경우 None을 리턴함

    Args:
        method (string): pipeline module
        input_type (string): 데이터 인풋타입, "DF" or "DFSet"

    Returns:
        String : output_type (데이터 output 타입, "DF" or "DFSet"/ or None)
    
    """
    
    
    rule = pipeline_rule[method]
    if input_type in rule.keys():
        output_type = rule[input_type]
    else:
        output_type = None
        
    return output_type 


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
    
def pipeline_result_EDA(data, module_name):
    """
    For EDA 
    
    Args:
        data() : data
        module_name(string) : module name
    
    """
    # For EDA
    import math
    import matplotlib.pyplot as plt

    def _plot_data(data, module_name):
        plt.figure(figsize=(15, 5))
        plt.title(module_name)
        plt.plot(data)
            

    def _plot_dataset(dataset, module_name):
        row = math.ceil(len(dataset)/2)
        plt.figure(figsize=(20,row*3.5))
        
        for i, data_name in enumerate(dataset):
            data = dataset[data_name]
            plt.rc('font', size=8)
            plt.subplot(row, 2, i+1)
            plt.title("{}_{}".format(data_name, module_name))
            plt.plot(data)

            
    def _count_nan_dataset(dataset):
        previous_nan = 0 
        previous_leng = 0 
        for i, data_name in enumerate(dataset):
            data = dataset[data_name]
            current_nan = data.isna().sum()
            previous_nan = current_nan + previous_nan
            previous_leng = len(data)+ previous_leng
            
                
        print("data_length:", previous_leng)
        print("Feature NaN number : ", previous_nan)
        
        
    def _count_nan(data):
        print("data_length:", 1)
        #print("Feature NaN number  : ", data.isna().sum())
        print("data_length:", len(data))
        print("All NaN number  : ", data.isna().sum().sum())
            
    def _plot_interface(data, module_name):
        if isinstance(data, dict):
            _plot_dataset(data, module_name)
        elif isinstance(data, pd.DataFrame):
            _plot_data(data, module_name)
            
    def _count_nan_interface(data):
        if isinstance(data, dict):
            _count_nan_dataset(data)
        elif isinstance(data, pd.DataFrame):
            _count_nan(data)
                    
    
    ##############################################################
    
    _plot_interface(data, module_name)
    _count_nan_interface(data)

