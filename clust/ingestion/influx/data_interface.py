import sys
sys.path.append("../")
sys.path.append("../../")

from Clust.clust.ingestion.influx import df_data, df_set_data

def get_data_result(ingestion_type, db_client, param) : 
        """조건에 맞게 데이터를 정합함

        Args:
            ingestion_type (string): ["multi_ms_integration", "multiMs_MsinBucket"]
            db_client (db_client): influxDB에서 데이터를 인출하기 위한 client
            param (_type_):ingestion_type에 따른 인출을 위해 필요한 parameter

        Returns:
            pd.DataFrame or dictionary of pd.DataFrame : 단일 dataframe 혹은 dataframe을 value로 갖는 dictionary
        """
        
        # data_param 하위에 'feature_list' key가 유효한 경우 한번더 필터링
        df_out_list = ['multi_ms_integration']
        df_set_out_list = ['multi_ms_one_enumerated_ms_in_bucket_integration']
        
        
        if ingestion_type in df_out_list:
            result = df_data.dfData(db_client).get_result(ingestion_type, param)
        elif ingestion_type in df_set_out_list:
            result = df_set_data.DfSetData(db_client).get_result(ingestion_type, param)
        
        return result
