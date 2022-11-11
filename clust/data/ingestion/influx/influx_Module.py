import pandas as pd
def getAllMSDataSetFromInfluxDB(start_time, end_time, db_client, db_name):
    """
        It returns dataSet from all MS of a speicific DB.

        :param db_client: instance to get data from influx DB
        :type db_client: instance of influxClient class
        :param db_name: db_name
        :type db_name: str
        :param query_start_time: query_start_time
        :type query_start_time: pd.datatime
        :param query_end_time: query_end_time
        :type query_end_time: pd.datatime

        :returns: dataSet
        :rtype: list of dataframe (ms datasets)
    """
    ms_list = db_client.measurement_list(db_name)
    dataSet ={}
    for ms_name in ms_list:
        data = db_client.get_data_by_time(start_time, end_time, db_name, ms_name)
        dataSet[ms_name] = data
    return dataSet

def get_MeasurementDataSetOnlyNumeric(db_client, intDataInfo):
        """
        Get measurement Data Set according to the dbinfo
        Each function makes dataframe output with "timedate" index.

        :param intDataInfo: intDataInfo
        :type intDataInfo: dic

        :return: MSdataset
        :rtype: Dict

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

def checkNumericColumns(data, checkColumnList=None):
    """
    This function returns data by trnsforming the Numeric type colums specified in "checkColumnList". 
    If checkColumnList is None, all columns are converted to Numeric type.

        :param data: inputData
        :type data: dataFrame
        :param checkColumnList: db_name
        :type db_name: string array or None

        :returns: dataSet
        :rtype: dataType

    1. CheckColumnList==None : change all columns to numeric type
    2. CheckColumnList has values: change only CheckColumnList to numeric type
    """
    if checkColumnList:
        pass
    
    else:
        checkColumnList = list(data.select_dtypes(include=['object']).columns)

    data[checkColumnList] = data[checkColumnList].apply(pd.to_numeric, errors='coerce')
    
    return data

def saveDataToInfluxDB(db_client, data):
    bk_name =''
    ms_name =''
    data_frame = data


    db_client.write_db(bk_name, ms_name, data_frame)
    db_client.close_db()




