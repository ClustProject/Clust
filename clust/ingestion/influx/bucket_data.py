import pandas as pd

def get_all_msData_in_bucketList(bucket_list, db_client, start_time, end_time, new_bucket_list = None):

    """
    - get all ms dataset in bucket_list (duration: start_time ~ end_time)
    - if new_bucket_list is not None,  change bucket_list name

    Args:
        bucket_list (string array): array of multiple bucket name 
        db_client: influx DB client
        start_time(datetime): start time of Data
        end_time(datetime): end time of Data
        new_bucket_list: default =None, new bucket name list
            example>>
                start_time = pd.to_datetime("2021-09-12 00:00:00")
                end_time = pd.to_datetime("2021-09-19 00:00:00")
                bucket_list =['air_indoor_요양원', 'air_indoor_아파트', 'air_indoor_체육시설', 'air_indoor_고등학교','air_indoor_도서관','air_indoor_경로당','air_indoor_유치원','air_indoor_어린이집','air_indoor_중학교','air_indoor_초등학교']
                new_bucket_list = ['air_indoor_yoyangwon','air_indoor_apartment','air_indoor_gym',  'highschool', 'library', 'seniorCenter','kindergarten','childcare','middleSchool', 'elementarySchool']
            
    Return:
        dataSet(dict of pd.DataFrame): new DataSet : key name  ===> msName + bucketName
    """
    data_set = {}
    if new_bucket_list is None:
        new_bucket_list = bucket_list

    for idx, bucket_name in enumerate(bucket_list):
        # data exploration start
        dataSet_indi = get_all_msData_in_oneBucket(start_time, end_time, db_client, bucket_name)
        print(bucket_name, " length:", len(dataSet_indi))
        dataSet_indi = {f'{k}_{new_bucket_list[idx]}': v for k, v in dataSet_indi.items()}
        data_set.update(dataSet_indi)

    return data_set


def get_all_msData_in_oneBucket(start_time, end_time, db_client, bucket_name):
    """
    It returns dataSet from all MS of a speicific DB(Bucket) from start_time to end_time

    Args:
        db_client: influx DB client
        start_time(datetime): start time of Data
        end_time(datetime): end time of Data
        db_client (instance of influxClient class): instance to get data from influx DB
        bucket_name (string): bucket name
    
    Returns:
        dataSet (dictionary): returned dataset ----> {"data1_name":DF1, "data2_name:DF2......}
    """

    ms_list = db_client.measurement_list(bucket_name)
    dataSet ={}
    for ms_name in ms_list:
        data = db_client.get_data_by_time(start_time, end_time, bucket_name, ms_name)
        if len(data)>0:
            dataSet[ms_name] = data

    return dataSet
