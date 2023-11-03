import os
import sys
sys.path.append("../")
sys.path.append("../../")

from Clust.clust.analysis import dataAnalysis, dataSetAnalysis
import pandas as pd
def get_analysis_result(analysis_method, analysis_param, input_data):
    """
    input에 따른 분석 결과를 도출하기 위해 두개의 함수로 분기하는 함수

    Args:
        analysis_method (_str_) : 분석 방법    
        >>> ["original", 'correlation', 'scaling', 'max_correlation_value_index_with_lag','scale_xy_frequency'] # dataframe input
        >>> ['multiple_maxabs_correlation_value_table_with_lag', 'multiple_maxabs_correlation_index_table_with_lag'] # dictionary input  

        analysis_param (_dict_) : analysis method에 따른 적절한 파라미터
        >>> analysis_param = {} #original, scaling
        >>> analysis_param = {'feature_key': 'PM10', 'lag_number': '24'} # max_correlation_value_index_with_lag, multiple_maxabs_correlation_value_table_with_lag, multiple_maxabs_correlation_index_table_with_lag   
        >>> analysis_param = {'lag_number': 24,'feature_key': 'CO2',
                            'time_scale': {'x_frequency': {'unit': 'H', 'num': 1}, 'y_frequency': {'unit': 'D', 'num': 1}}} # scale_xy_frequency 
        >>>  input_data (_pd.dataFrame_ or _dict(pd.dataFrame)_) : 두가지 input type이 있을 수 있으며, analysis_method에 따라 input type은 고정됨

    Returns:
        pd.dataFrame : df_analysis, 분석 결과

    """
    
    # 가능한 분석 기법
    analysis_by_data_list       = ["original", 'correlation', 'scaling', 'max_correlation_value_index_with_lag','scale_xy_frequency'] # dataframe input
    analysis_by_data_set_list   = ['multiple_maxabs_correlation_value_table_with_lag', 'multiple_maxabs_correlation_index_table_with_lag'] # dictionary input
    
    if analysis_method in analysis_by_data_list:
        df_analysis = get_analysis_by_data(analysis_method, analysis_param, input_data)

    elif analysis_method in analysis_by_data_set_list:
        df_analysis = get_analysis_by_data_set(analysis_method, analysis_param, input_data)

        
    return df_analysis

def get_analysis_by_data(analysis_method, analysis_param, input_df):
    """    
    input이 dataframe일 경우 분석을 수행하는 함수

    Args:
        analysis_method (_str_) : analysis method
        analysis_param (_dict_) : analysis method에 따른 적절한 파라미터
        input_df (_pd.dataFrame_) : 분석에 필요한 인풋 데이터

    Returns:
        _pd.dataframe_ : df_analysis, 분석 결과     

    """
    
    da = dataAnalysis.DataAnalysis()
    if analysis_method == 'original':
        df_analysis = input_df

    elif analysis_method == 'correlation':
        df_analysis = input_df.corr()
        df_analysis = round(df_analysis, 2)

    elif analysis_method == 'scaling':
        from Clust.clust.transformation.general import basicTransform
        df_analysis = basicTransform.getRobustScaledDF(input_df)

    elif analysis_method == 'max_correlation_value_index_with_lag':
        df_analysis = da.get_max_correlation_table_with_lag(analysis_param, input_df) 
        print(df_analysis)
        
    elif analysis_method == 'scale_xy_frequency':
        time_scale = analysis_param['time_scale']
        df_analysis = da.scale_different_x_y_frequency(time_scale, input_df) 
                
    return df_analysis

def get_analysis_by_data_set(analysis_method, analysis_param, input_df_set):
    """ 
    analysis_method가 'multiple_maxabs_correlation_value_table_with_lag', 'multiple_maxabs_correlation_index_table_with_lag' 둘 중 하나의 경우 분석 수행

    Args:
        analysis_method (_str_) : 분석 방법
        >>> ['multiple_maxabs_correlation_value_table_with_lag', 'multiple_maxabs_correlation_index_table_with_lag']
    
        analysis_param (_dict_) : analysis method에 따른 적절한 파라미터
        >>> 'analysis_param': {'feature_key': 'PM10', 'lag_number': '24'}
    
        input_df (_pd.dataFrame_) : 분석에 필요한 인풋 데이터

    Returns:
        pd.dataframe : df_analysis, 분석 결과
        
    TODO: 각 analysis_method에 따른 파라미터 예제 모두 기입할 것
    
    """

    dsa = dataSetAnalysis.DataSetAnalysis()
    if analysis_method == 'multiple_maxabs_correlation_value_table_with_lag':
        df_analysis = dsa.get_multiple_max_correlation_value_table_with_lag(analysis_param, input_df_set)       
        
    elif analysis_method == 'multiple_maxabs_correlation_index_table_with_lag':
        df_analysis = dsa.get_multiple_max_correlation_index_table_with_lag(analysis_param, input_df_set)
        
    
    df_analysis = df_analysis.apply(pd.to_numeric).abs()
    
    return df_analysis
