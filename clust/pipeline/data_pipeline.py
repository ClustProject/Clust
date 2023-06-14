import sys
sys.path.append("../")
sys.path.append("../..")
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
    }
}

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


def pipeline_connection_check(pipeline, input_type):
    """pipeline의 in-output module 결합 유효성을 검사한다.

    Args:
        module1 (string): 선모듈 이름
        module2 (string): 후모듈 이름
        input_type (string): 데이터 인풋타입, "DF" or "DFSet"
        
     Returns:
        output_type(Bool): True or False, 유효성 여부 전달
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
        print(method, input_type, output_type)
    return valid

def pipeline_module_check(method, input_type):
    """pipeline module의 인풋과 메소드가 유효한지를 체크함, 유효한 경우 output type을 뱉고 유효하지 않을 경우 None을 리턴함

    Args:
        method (string): pipeline module
        input_type (string): 데이터 인풋타입, "DF" or "DFSet"

    Returns:
        output_type(string): 데이터 output 타입, "DF" or "DFSet"/ or None
    
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
