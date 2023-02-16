import pandas as pd

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
    
def get_integated_multi_ms(data_param, db_client):
    """get integrated numeric data with multiple MS data

    Args:
        data_param (_type_): data_param 
        db_client (_type_): influx db client

    Returns:
        _type_: _description_
    """
    
    ms_array        = data_param['ms_array']
    start_time          = data_param['start_time']
    end_time            = data_param['end_time']
    integration_freq_sec = int(data_param['integration_freq_min']) * 60 
    
    #TODO 추후 이부분 외부 입력 받도록
    integration_param = get_integration_param(integration_freq_sec)
    process_param = get_general_process_param()
    

    from Clust.clust.integration.utils import param
    intDataInfo = param.makeIntDataInfoSet(ms_array, start_time, end_time) 
    from Clust.clust.ingestion.influx import ms_data
    multiple_dataset  = get_only_numericData_in_ms(db_client, intDataInfo)  
    
    # data Integration
    from Clust.clust.integration.integrationInterface import IntegrationInterface
    dataIntegrated = IntegrationInterface().multipleDatasetsIntegration(process_param, integration_param, multiple_dataset)
    
    return dataIntegrated


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
