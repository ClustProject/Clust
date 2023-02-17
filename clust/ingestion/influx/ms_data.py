import pandas as pd

def get_data_by_ingestion_type(data_ingestion_type, db_client, data_ingestion_param) : 
        """조건에 맞게 데이터를 정합함

        Args:
            data_ingestion_type (string): ["multiMS", "multiMs_MsinBucket"]
            db_client (db_client): influxDB에서 데이터를 인출하기 위한 client
            data_ingestion_param (_type_):data_ingestion_type에 따른 인출을 위해 필요한 parameter

        Returns:
            pd.DataFrame or dictionary of pd.DataFrame : 단일 dataframe 혹은 dataframe을 value로 갖는 dictionary
        """
        
        # data_param 하위에 'feature_list' key가 유효한 경우 한번더 필터링

        # Step 1. get all dataset
        if data_ingestion_type == "multiMS":
            # result(dataframe)
            result = get_integated_multi_ms(data_ingestion_param, db_client)
            
        elif data_ingestion_type == "multiMs_MsinBucket":
            # result(dictionary of dataframe value)
            result = get_integated_multi_ms_and_one_bucket(data_ingestion_param, db_client)
        else:
            # result(dataframe)
            result = get_integated_multi_ms(data_ingestion_param, db_client)
        
            
        return result

def get_integated_multi_ms(data_param, db_client):
    """ get integrated numeric data with multiple MS data

    Args:
        data_param (dict): data_param 
        db_client (db_client): influx db client

    >>> data_param = {
        'ms_list_info':,
        'start_time':,
        'end_time':,
        'integration_freq_min:,
        'feature_list:[]
    }
    Returns:
        pd.dataFrame: integrated data
    """
    
    ms_list_info        = data_param['ms_list_info']
    start_time          = data_param['start_time']
    end_time            = data_param['end_time']
    integration_freq_sec = int(data_param['integration_freq_min']) * 60 
    
    #TODO 추후 이부분 외부 입력 받도록
    integration_param = get_integration_param(integration_freq_sec)
    process_param = get_general_process_param()
    

    from Clust.clust.integration.utils import param
    intDataInfo = param.makeIntDataInfoSet(ms_list_info, start_time, end_time) 
    from Clust.clust.ingestion.influx import ms_data
    multiple_dataset  = get_only_numericData_in_ms(db_client, intDataInfo)  
    
    # data Integration
    from Clust.clust.integration.integrationInterface import IntegrationInterface
    dataIntegrated = IntegrationInterface().multipleDatasetsIntegration(process_param, integration_param, multiple_dataset)
    
    if data_param['feature_list']:
        dataIntegrated = dataIntegrated[data_param['feature_list']]
    
    return dataIntegrated

def get_integated_multi_ms_and_one_bucket(data_param, db_client):
    """1개의 특정 bucket에 있는 모든 ms (multiple ms in bucket) 와 고정된 다른 ms (ms_list_info) 들을 복합하여 데이터를 준비함, feature_list 가 명시되었다면 명시된 feature_list와 관련한 데이터만 전달

    Args:
        data_param (dict): data_param 
        db_client (db_client): influx db client

    >>> data_param = {
        'data_org':,
        'bucket_name': , 
        'start_time':,
        'end_time':,
        'integration_freq_min:,
        'feature_list:[]
    }
    
    Returns:
        pd.dataFrame: integrated data
    """
    
    data_org        = data_param['data_org']
    bucket_name         = data_param['bucket_name']
    start_time          = data_param['start_time']
    end_time            = data_param['end_time']
    integration_freq_sec = int(data_param['integration_freq_min']) * 60 
    bucket_dataSet={}
    
    ms_list = db_client.measurement_list(bucket_name) #ms_name
    for ms_name in ms_list:
        dataInfo = data_org
        dataInfo = data_org + [[bucket_name, ms_name]] 
        data_param['ms_list_info'] = dataInfo
        dataIntegrated = get_integated_multi_ms(data_param, db_client)
        if data_param['feature_list']:
            dataIntegrated = dataIntegrated[data_param['feature_list']]
        bucket_dataSet[ms_name]= dataIntegrated
    
    return bucket_dataSet

def get_general_process_param():
    refine_param        = {"removeDuplication":{"flag":True},"staticFrequency":{"flag":True, "frequency":None}}
    CertainParam        = {'flag': True}
    uncertainParam      = {'flag': False, "param": {"outlierDetectorConfig":[{'algorithm': 'IQR', 'percentile':99 ,'alg_parameter': {'weight':100}}]}}
    outlier_param       = {"certainErrorToNaN":CertainParam, "unCertainErrorToNaN":uncertainParam}
    imputation_param    = {
        "flag":True,
        "imputation_method":[{"min":0,"max":3,"method":"linear", "parameter":{}}],
        "totalNonNanRatio":80
    }

    # 최종 파라미터
    process_param       = {'refine_param':refine_param, 'outlier_param':outlier_param, 'imputation_param':imputation_param}
    return process_param

def get_integration_param(integration_freq_sec):
    integration_param   = {
        "integration_duration":"common",
        "integration_frequency":integration_freq_sec,
        "param":{},
        "method":"meta"
    }
    return integration_param


def get_only_numericData_in_ms(db_client, intDataInfo):
    """
    Get measurement Data Set according to the dbinfo
    Each function makes dataframe output with "timedate" index.

    Args:
        db_client (string): db_client (instance of influxClient class): instance to get data from influx DB
        intDataInfo (dict): intDataInfo
            example>>> intDataInfo = {"db_info":
                       [{"db_name":"INNER_AIR","measurement":"HS1","start":str(start),"end":str(end)},
                     {"db_name":"OUTDOOR_AIR","measurement":"sangju","start":str(start),"end":str(end)},
                     {"db_name":"OUTDOOR_WEATHER","measurement":"sangju","start":str(start),"end":str(end)}]}

    
    Returns:
        Dictionary: MSdataset
    """
    

    MSdataSet ={}
    for i, dbinfo in enumerate(intDataInfo['db_info']):
        db_name = dbinfo['db_name']
        ms_name = dbinfo['measurement']
        tag_key =None
        tag_value =None 
        if "tag_key" in dbinfo.keys():
            if "tag_value" in dbinfo.keys():
                tag_key = dbinfo['tag_key']
                tag_value = dbinfo['tag_value']

        import numpy as np
        multiple_dataset=db_client.get_data_by_time(dbinfo['start'], dbinfo['end'], db_name, ms_name, tag_key, tag_value)
        MSdataSet[i]  =  multiple_dataset.select_dtypes(include=np.number)
        MSdataSet[i].index.name ='datetime'

    

    return MSdataSet
