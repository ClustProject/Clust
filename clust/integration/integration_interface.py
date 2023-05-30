import sys
sys.path.append("../")
sys.path.append("../..")
sys.path.append("../../..")
import pandas as pd 

def get_data_result(integration_type, data_set, integration_param):
    """get integrated data by integration_type and parameter

    Args:
        integration_type (string): integration type ['one_feature_based_integration',  'multiple_dataset_integration']
        data_set (dictionary of dataframe): data input 
        integration_param (dictionary): integratoin parameter

    Returns:
        dataframe: integrated dataframe
    """
    
    if integration_type =='one_feature_based_integration':
        feature_name = integration_param['feature_name']
        if 'duration' in list(integration_param.keys()):
            duration = integration_param['duration']
        else:
            duration = None
            
        if 'frequency' in list(integration_param.keys()):
            frequency = integration_param['frequency']
        else:
            frequency = None
            
        result = get_one_feature_based_integration(data_set, feature_name, duration, frequency)
        
    if integration_type =="multiple_dataset_integration":
        from Clust.clust.integration.integrationInterface import IntegrationInterface
        result = IntegrationInterface().multipleDatasetsIntegration(integration_param, data_set)
    
    return result

def get_one_feature_based_integration(dataSet, feature_name, duration, frequency):
    """
    1) choose only one column value from each data of dataSet
    2) make one dataFrame.
    
    Args:
        dataSet (dictionary of DataFrame) : key = dataName, value = data(dataFrame) (Each data has must same length)
        feature_name(string): feature name of data
        durtaion (dict):
        frequency (timedelta):
        
    Return:
        newDF (dataFrame): new DataFrame
    """
    def _get_multipleDF_sameDuration(data, duration, frequency):
        # Make Data with Full Duration [query_start_time ~ query_end_time]
        
        start_time =duration['start_time']
        end_time = duration['end_time']

        if len(data)>0:
            #2. Make Full Data(query Start ~ end) with NaN
            data.index = data.index.tz_localize(None) # 지역정보 없이 시간대만 가져오기
            new_idx = pd.date_range(start = start_time, end = (end_time- frequency), freq = frequency)
            new_data = pd.DataFrame(index= new_idx)
            new_data.index.name ='time' 
            data = new_data.join(data) # new_data에 data를 덮어쓴다
        return data
    
    newDF = pd.DataFrame()
    for data_name in dataSet:
        data = dataSet[data_name]
        if duration:
            data = _get_multipleDF_sameDuration(data, duration, frequency)

        if feature_name in list(data.columns):
            if len(data) > 0:
                newDF[data_name] = data[feature_name].values # 이 부분 문제 생길수 있음

    return newDF

