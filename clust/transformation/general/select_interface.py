import sys
sys.path.append("../")
sys.path.append("../../")
import pandas as pd

def get_data_result(select_type, data_input, select_param=None):
    """
    select data by select_type and parameter

    Args:
        select_type (string):select type 
        data_input (dictionary) :original data set
        param: select parameter
    
    Returns:
        dictionary: Return selected dictionary dataset
        
        {'keyword': '/afternoon'}
    select_param example:
        1. select_type: keyword
        >>> select_param = {'keyword': '/afternoon'}
        
    """
    
    if select_type =='keyword_data_selection':
        keyword = select_param['keyword']
        old_keys = list(set(data_input.keys()))
        new_keys = [e for e in old_keys if keyword in e]
        result = dict((k, data_input[k]) for k in new_keys if k in data_input)
    
    if select_type =='oneDF_with_oneFeature_from_multipleDF':
        feature_name = select_param['feature_name']
        if select_param['duration']:
            duration = select_param['duration']
        else:
            duration = None
        if select_param['frequency']:
            frequency = select_param['frequency']
        else:
            frequency = None
        result = get_oneDF_with_oneFeature_from_multipleDF(data_input, feature_name, duration, frequency)

    return result

def get_oneDF_with_oneFeature_from_multipleDF(dataSet, feature_name, duration, frequency):
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
            newDF[data_name] = data[feature_name].values # 이 부분 문제 생길수 있음

    return newDF

