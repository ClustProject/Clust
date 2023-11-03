import sys
sys.path.append("../")
sys.path.append("../../")

from Clust.clust.data import df_data, df_set_data

def get_data_result(ingestion_type, db_client, param) : 
    """ 
    - Get data by condition

    Args:
        ingestion_type (_String_) : Ingestion type is as follows.
        db_client (_instance_) : Instance of InfluxClient class. Instance to get data from influx DB.
        param (_Dictionary_) : Parametes required for extraction according to ingestion_type, each different depending on the ingestion type.

    **Ingestion Type**::
    
        There are several ingestion types based on dataframe and dataframe set.
        * For dataframe: ms_by_num, ms_by_days, ms_by_time, ms_all
        * For dataframe Set: all_ms_in_one_bucket, all_ms_in_multiple_bucket,
                            multiple_ms_by_time, multi_ms_one_enumerated_ms_in_bucket_integration
                  
    
    Returns:
        Dataframe or Dictionary : result(A single dataframe or a dictionary with dataframe as value.)

    >>> ingestion_type = 'ms_by_num'
                
    """
        # data_param에 'feature_list' key가 유효한 경우 한번 더 필터링
        
    df_out_list     = ['ms_by_num', 'ms_by_days', 'ms_by_time', 'ms_all']
    df_set_out_list = ['multiple_ms_by_time',
                        'multi_ms_one_enumerated_ms_in_bucket_integration',           
                        'all_ms_in_one_bucket', 
                        'all_ms_in_multiple_bucket']        
    
    if ingestion_type in df_out_list:
        result = df_data.DfData(db_client).get_result(ingestion_type, param)
        
    elif ingestion_type in df_set_out_list:
        result = df_set_data.DfSetData(db_client).get_result(ingestion_type, param)

        
    return result
