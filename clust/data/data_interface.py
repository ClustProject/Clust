import sys
sys.path.append("../")
sys.path.append("../../")

from Clust.clust.data import df_data, df_set_data

def get_data_result(ingestion_type, db_client, param) : 
        """ 
        # Description
        조건에 맞게 데이터를 정합함

        # Args
         * ingestion_type (_str_)
        ```example        
        
            >>> ingestion_type = ['multi_ms_integration', 'ms_by_days', 'ms_by_time', 'ms_by_num']  
        ```
         * db_client (_db_client_) : influxDB에서 데이터를 인출하기 위한 client
         * param (_dict_) : ingestion_type에 따른 인출을 위해 필요한 parameter
        ```example
        
           
        ```

        # Returns
         * result (_pd.DataFrame_ or _dict of pd.DataFrame_) : 단일 dataframe 혹은 dataframe을 value로 갖는 dictionary

        """
        
        # data_param 하위에 'feature_list' key가 유효한 경우 한번 더 필터링
        df_out_list     = ['ms_by_num', 'ms_by_days', 'ms_by_time', 'multi_ms_integration']
        df_set_out_list = ['multi_ms_one_enumerated_ms_in_bucket_integration',
                           'multi_numeric_ms_list',
                           'all_ms_in_one_bucket', 'all_ms_in_multiple_bucket']        
        
        if ingestion_type in df_out_list:
            result = df_data.DfData(db_client).get_result(ingestion_type, param)
            
        elif ingestion_type in df_set_out_list:
            result = df_set_data.DfSetData(db_client).get_result(ingestion_type, param)
        
        return result
