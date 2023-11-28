import pandas as pd
from . import dataAnalysis

class DataSetAnalysis():
    def get_column_list (self, df_set):
        """        
        여러 데이터셋에 대한 공통 컬럼 리스트를 뽑아냄

        Args:
            df_set (_Dictionary(pd.dataFrame)_)

        Returns:
            array(string) : column_list

        """
        
        column_list = next(iter((df_set.items())))[1].columns
        return column_list
    
    def max_correlation_timelag(self, analysis_param, df_set, column_flag):
        """       
        lag를 적용하여 상관관계를 구한 후 가장 높은 값의 value table 추출

        Args:
            analysis_param (_Dictionary_)
            df_set (_Dictionary(pd.dataFrame)_)
            column_flag :['index'|'value']

        Returns:
            pd.dataFrame : max_correlation_timelag 

        """
        column_list = self.get_column_list (df_set)
        max_correlation_timelag = pd.DataFrame(index = column_list)

        for df_name in df_set.keys():
            data = df_set[df_name]
            max_position_table = dataAnalysis.DataAnalysis().get_max_correlation_table_with_lag(analysis_param, data)
            max_correlation_timelag[df_name] = max_position_table[column_flag]

        return max_correlation_timelag
    
    def get_multiple_max_correlation_value_table_with_lag(self, analysis_param, df_set):
        """        
        lag를 적용하여 상관관계를 구한 후 가장 높은 값의 value table 추출

        Args:
            analysis_param (_Dictionary_)
            df_set (_Dictionary(pd.dataFrame)_)

        Returns:
            pd.dataFrame : max_correlation_value_timelag

        """
        column_flag = 'value'
        max_correlation_value_timelag = self.max_correlation_timelag(analysis_param, df_set, column_flag)
        return max_correlation_value_timelag
    
    def get_multiple_max_correlation_index_table_with_lag(self, analysis_param, df_set):
        """        
        lag를 적용하여 상관관계를 구한 후 가장 높은 값의 index table 추출

        Args:
            analysis_param (_Dictionary_)
            df_set (_Dictionary(pd.dataFrame)_)

        Returns:
            pd.dataFrame : max_correlation_index_timelag
            
        """
        column_flag = 'index'
        max_correlation_index_timelag = self.max_correlation_timelag(analysis_param, df_set, column_flag)

        return max_correlation_index_timelag
    
    ####Analysis #########################################################
    