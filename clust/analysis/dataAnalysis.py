def get_analysis_by_data(analysis_method, analysis_param, df):
    """_summary_

    Args:
        anaylsis_method (_type_): _description_
        analysis_param (_type_): 각 analysis_method에 따라 서로 다른 형태의 key 값을 가짐
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    da = dataAnalysis()
    if analysis_method == 'origirnal':
        df_analysis = df
    elif analysis_method == 'correlation':
        df_analysis = df.corr()
    elif analysis_method == 'max_correlation_value_index_with_lag':
        """sumary_line
        analysis_param = {
            "feature_key":"string", # 기준으로 잡는 feature
            "lag_number": int # 조사하는 lag 개수
        }
        
        """
        df_analysis = da.get_max_correlation_table_with_lag(analysis_param, df)
        
    return df_analysis

def get_analysis_by_data_set(analysis_method, analysis_param, df_set):
    """_summary_

    Args:
        anaylsis_method (_type_): _description_
        analysis_param (_type_): 각 analysis_method에 따라 서로 다른 형태의 key 값을 가짐
        df_set (_type_): _description_

    Returns:
        _type_: _description_
    """
    da = dataAnalysis()
    if analysis_method == 'multiple_maxabs_correlation_value_table_with_lag':
        """sumary_line
        analysis_param = {
            "feature_key":"string", # 기준으로 잡는 feature
            "lag_number": int # 조사하는 lag 개수
        }
        
        """
        df_analysis = da.get_multiple_max_correlation_value_table_with_lag(analysis_param, df_set)
        # 출력은 절대값으로
        df_analysis = df_analysis.apply(pd.to_numeric)
        
    elif analysis_method == 'multiple_maxabs_correlation_index_table_with_lag':
        """sumary_line
        analysis_param = {
            "feature_key":"string", # 기준으로 잡는 feature
            "lag_number": int # 조사하는 lag 개수
        }
        
        """
        df_analysis = da.get_multiple_max_correlation_index_table_with_lag(analysis_param, df_set)
        df_analysis = df_analysis.apply(pd.to_numeric).abs()
    
        
    return df_analysis

import pandas as pd
class dataAnalysis():
    ####Analysis #########################################################
    ##추후 analysis는 다른쪽으로 (다른 모듈, 다른 클래스) 빼는게 맞아 보임
    def get_max_correlation_table_with_lag(self, analysis_param, df):
        """_summary_

        Args:
            analysis_method (_type_): _description_
            df (_type_): _description_

        Returns:
            _type_: _description_
        """
        feature_key = analysis_param['feature_key']
        lag_number = analysis_param['lag_number']
        # feature_key, lag_number
        
        from Clust.clust.tool.stats_table import timelagCorr
        CCT = timelagCorr.TimeLagCorr()
        result = CCT.df_timelag_crosscorr(df, feature_key, lag_number)
        max_position_correlation_table = CCT.get_absmax_index_and_values(result) 
        
        return max_position_correlation_table
    
    def get_multiple_max_correlation_value_table_with_lag(self, analysis_param, df_set):
        column_list = next(iter((df_set.items())))[1].columns
        max_correlation_value_timelag = pd.DataFrame(index = column_list)
        feature_key = analysis_param['feature_key']
        lag_number = analysis_param['lag_number']

        for df_name in df_set.keys():
            data = df_set[df_name]
            
            #################################################
            max_position_table = self.get_max_correlation_table_with_lag(analysis_param, data)
            max_correlation_value_timelag[df_name]=max_position_table['value']

        return max_correlation_value_timelag
    
    def get_multiple_max_correlation_index_table_with_lag(self, analysis_param, df_set):
        column_list = next(iter((df_set.items())))[1].columns
        max_correlation_index_timelag = pd.DataFrame(index = column_list)
        feature_key = analysis_param['feature_key']
        lag_number = analysis_param['lag_number']
        for df_name in df_set.keys():
            data = df_set[df_name]
            #################################################
            max_position_table = self.get_max_correlation_table_with_lag(analysis_param, data)
            max_correlation_index_timelag[df_name]=max_position_table['index']

        return max_correlation_index_timelag
    
    ####Analysis #########################################################
    