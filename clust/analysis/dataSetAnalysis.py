import pandas as pd
class DataSetAnalysis():
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
    