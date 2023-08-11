import pandas as pd

def get_data_frame(param, db_client = None):
    """Parameter에 따라 데이터를 인출해옴

    Args:
        param (dict): 데이터를 인출하기 위한 파라미터

    Returns:
        data: 인출된 파라미터
    """
    data_type = param['data_type']
    param = param['param']
    if data_type == 'csv':
        bucket_name = param['bucket_name']
        ms_name = param['ms_name']
        data = db_client.get_data(bucket_name, ms_name)
    
    elif data_type =='influx':
        file_path = param['file_path']
        data = pd.read_csv(file_path, index_col=0)
        
    return data

def save_data_frame(data, param, db_client = None):
    """data를 influx 혹은 CSV로 저장함

    Args:
        data(pd.DataFrame): data
        param (dict): parameter for data savding
        db_client (influx client, optional): influx information. Defaults to None.
    """
    
    data_type = param['data_type']
    param = param['param']
    
    if data_type == 'csv':
        file_path = param['file_path']
        from Clust.clust.ingestion.DataToCSV import dfToCSV
        dfToCSV.save_data(data, file_path)
    elif data_type =='influx':
        bucket_name = param['bucket_name']
        measurement_name = param['measurement_name']
        db_client.write_db(bucket_name, measurement_name, data)
    
        
         
    