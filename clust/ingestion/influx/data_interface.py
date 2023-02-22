import sys
sys.path.append("../")
sys.path.append("../../")

from Clust.clust.ingestion.influx import dfData, dfSetData

def get_data_result(ingestion_type, db_client, param) : 
        """조건에 맞게 데이터를 정합함

        Args:
            ingestion_type (string): ["multiMS", "multiMs_MsinBucket"]
            db_client (db_client): influxDB에서 데이터를 인출하기 위한 client
            param (_type_):ingestion_type에 따른 인출을 위해 필요한 parameter

        Returns:
            pd.DataFrame or dictionary of pd.DataFrame : 단일 dataframe 혹은 dataframe을 value로 갖는 dictionary
        """
        
        # data_param 하위에 'feature_list' key가 유효한 경우 한번더 필터링
        df_out_list = ['multiMS']
        df_set_out_list = ['multiMs_MsinBucket']
        
        
        if ingestion_type in df_out_list:
            result = df_Data.dfData(ingestion_type, param, db_client)
        elif ingestion_type in df_set_out_list:
            result = df_Set_data.DfSetData(ingestion_type, param, db_client)
        
        return result
