import pandas as pd
def get_oneDF_with_oneFeature_from_multipleDF(dataSet, feature_name, duration=None, frequency = None):
    """
    1) choose only one column value from each data of dataSet a
    2) make one dataFrame.
    
    Args:
        dataSet (dictionary of DataFrame) : key = dataName, value = data(dataFrame)
        feature_name(string): feature name of data
        
    Return:
        newDF (dataFrame): new DataFrame
    """
    newDF = pd.DataFrame()
    for data_name in dataSet:
        data = dataSet[data_name]

        if duration:
            data = get_multipleDF_sameDuration(data, duration, frequency)

        if feature_name in list(data.columns):
            value = data[feature_name].values
            newDF[data_name] = value
    return newDF


# self.query_start_time, self.query_end_time 문제 이슈
def get_multipleDF_sameDuration(data, duration, frequency):
    # Make Data with Full Duration [query_start_time ~ query_end_time]
    
    start_time =duration['start_time']
    end_time = duration['end_time']
    # print("get_multipleDF_sameDuration", start_time, end_time)
    if len(data)>0:
        #2. Make Full Data(query Start ~ end) with NaN
        data.index = data.index.tz_localize(None) # 지역정보 없이 시간대만 가져오기
        new_idx = pd.date_range(start = start_time, end = (end_time- frequency), freq = frequency)
        new_data = pd.DataFrame(index= new_idx)
        new_data.index.name ='time' 
        data = new_data.join(data) # new_data에 data를 덮어쓴다
    return data

