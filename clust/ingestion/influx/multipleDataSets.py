import pandas as pd

def getAllMSDataSetFromInfluxDB(start_time, end_time, db_client, db_name):
    """
    It returns dataSet from all MS of a speicific DB(Bucket) from start_time to end_time

    Args:
        start_time (Timestamp): start time
        end_time (Timestamp): end time
        db_client (instance of influxClient class): instance to get data from influx DB
        db_name (string): database
    
    Returns:
        Dictionary: dataSet, list of dataframe (ms datasets)
    """

    ms_list = db_client.measurement_list(db_name)
    dataSet ={}
    for ms_name in ms_list:
        data = db_client.get_data_by_time(start_time, end_time, db_name, ms_name)
        dataSet[ms_name] = data

    return dataSet

def get_onlyNumericDataSets(db_client, intDataInfo):
    """
    Get measurement Data Set according to the dbinfo
    Each function makes dataframe output with "timedate" index.

    Args:
        db_client (string): db_client (instance of influxClient class): instance to get data from influx DB
        intDataInfo (dict): intDataInfo
    
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
        
        
        # TODO Only Numeric Data
        import numpy as np
        multiple_dataset=db_client.get_data_by_time(dbinfo['start'], dbinfo['end'], db_name, ms_name, tag_key, tag_value)
        MSdataSet[i]  =  multiple_dataset.select_dtypes(include=np.number)
        MSdataSet[i].index.name ='datetime'

    return MSdataSet
