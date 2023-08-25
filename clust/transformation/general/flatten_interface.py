import pandas as pd

def make_univariate_df(data, new_column_name):
    """transform multivariate data to univariate data 
    
    Args:
        data (pd.DataFrame): data frame with multiple columns
        new_column_name (str): new univariate column name

    Returns:
        result_df (pd.DataFrame): data frame with one column
    """
    result_df = pd.DataFrame()
    for name in data:
        print(data)
        result_df = pd.concat([result_df, data[name]])
    result_df.columns = [new_column_name]
    
    return result_df
    

def make_data_with_time_index(data, start_time, frequency):
    """add timeindex for data frame
    
    Args:
        data (pd.DataFrame): data frame 
        start_time(datetime): start time
        frequency(int): minutes, time series data description duration

    Returns:
        data (pd.DataFrame): data frame with time index (index name = time)
    """
    time_index = pd.date_range(start=start_time, freq = str(frequency)+"T", periods=len(data))
    data.set_index(time_index, inplace = True)
    data.index.name = "time"
    
    return data

def make_uni_variate_with_time_index(data, flatten_param):
    """"transform multivariate data to univariate data with time index
    
    Args:
        data (pd.DataFrame): data frame with multiple columns
        flatten_param(dictionary) : flattening param
    
     Example:
            >>> flatten_param :{
            ...     'start_time' : pd.to_datetime('2010-01-01 00:00:00'),
            ...     'frequency' : 1,
            ...     'new_column_name' : 'in_co2'
            ...     }

            >>> start_time (datetime) : new start time
            >>> frequency (int) : minutes
            >>> new_column_name(int or None) : new column name, option 
            
    Returns:
        result_df (pd.DataFrame): data frame with time index (index name = time)
    """
    start_time = flatten_param['start_time']
    frequency = flatten_param['frequency']
    new_column_name = flatten_param['new_column_name']

    if not new_column_name:
        new_column_name ="column1"
    
    data = make_univariate_df(data, new_column_name)
    result_df = make_data_with_time_index(data, start_time, frequency)

    return result_df
    