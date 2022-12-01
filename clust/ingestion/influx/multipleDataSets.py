import pandas as pd
def get_all_msdata_in_bucket_list(bucket_list, db_client, start_time, end_time, new_bucket_list = None):
    """
    - get all ms dataset by bucket_list
    - if new_bucket_list is not None,  change bucket_list name
    
    Args:
        db_client: influx DB client
        start_time: start time of Data
        end_time: end time of Data
        new_bucket_list: default =None, new bucket name list
    
    Return:
        dataSet(dict of pd.DataFrame): new DataSet
    """
    dataSet={}
    for idx, bucket_name in enumerate(bucket_list):
        # data exploration start
        print(bucket_name)
        dataSet_indi = get_all_msdata_in_bucket(start_time, end_time, db_client, bucket_name)
        if new_bucket_list:
            new_bucket_name = new_bucket_list[idx]
        dataSet_indi = {f'{k}_{new_bucket_name}': v for k, v in dataSet_indi.items()}
        dataSet.update(dataSet_indi)

    return dataSet


def get_all_msdata_in_bucket(start_time, end_time, db_client, bucket_name):
    """
        It returns dataSet from all MS of a speicific DB(Bucket) from start_time to end_time

        :param db_client: instance to get data from influx DB
        :type db_client: instance of influxClient class

        :param bucket_name: db_name
        :type bucket_name: str
        
        :param query_start_time: query_start_time
        :type query_start_time: pd.datatime
        
        :param query_end_time: query_end_time
        :type query_end_time: pd.datatime

        :returns: dataSet
        :rtype: list of dataframe (ms datasets)
    """
    ms_list = db_client.measurement_list(bucket_name)
    dataSet ={}
    for ms_name in ms_list:
        data = db_client.get_data_by_time(start_time, end_time, bucket_name, ms_name)
        dataSet[ms_name] = data

    return dataSet

def get_onlyNumericDataSets(db_client, intDataInfo):

        """
        Get measurement Data Set according to the dbinfo
        Each function makes dataframe output with "timedate" index.

        :param intDataInfo: intDataInfo
        :type intDataInfo: dic

        :return: MSdataset
        :rtype: Dictionary

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
