import pandas as pd
####Analysis #########################################################
##추후 analysis는 다른쪽으로 (다른 모듈, 다른 클래스) 빼는게 맞아 보임

class DataAnalysis():
    
    def get_max_correlation_table_with_lag(self, analysis_param, df):
        """
        # Description

        # Args
         * analysis_method (_pd.dataFrame_)
         ```example
            'analysis_param': {'feature_key': 'PM10', 'lag_number': '24'}
         ```
         * df (_pd.dataFrame_)

        # Returns
         * max_position_correlation_table (_type_) 

        """

        feature_key = analysis_param['feature_key']
        lag_number  = analysis_param['lag_number']
        # feature_key, lag_number
        
        from Clust.clust.tool.stats_table import timelagCorr
        CCT = timelagCorr.TimeLagCorr()
        result = CCT.df_timelag_crosscorr(df, feature_key, lag_number)
        max_position_correlation_table = CCT.get_absmax_index_and_values(result) 
        
        return max_position_correlation_table
   