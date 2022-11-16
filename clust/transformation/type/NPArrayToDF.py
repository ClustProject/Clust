import pandas as pd
from datetime import timedelta

def trans3NPtoDF(X, y, startTime):
    """
    Make Dataframe by input data condition.
    Every Sample of X has (featureNum X sequenceNum) datapoints.
    Every Sample of y has (one value) related to the value of X.
    Every New Sample can be separated by time index (Every Days)
    - X.shape (sampleNum, featureNum, sequenceNum )
    - y.shape (sampleNum, )

    :param: X
    :type: 3D numpy array
    
    :param: y
    :type: 1D numpy array

    :param: startTime: start Time of DataFrame Index
    :type: string

    :return: df_X, df_y
    :type: DataFrame
    
    """
    sampleNum = X.shape[0]
    featureNum = X.shape[1]
    sequenceNum = X.shape[2]

    interval = timedelta(days=1)
    duration = interval/sequenceNum

    df_X = pd.DataFrame()
    df_y = pd.DataFrame(columns=['value'])
    timeIndex1List = pd.date_range(startTime,freq =interval, periods =sampleNum)

    for i, timeIndex1 in enumerate(timeIndex1List):
        data_x = X[i]
        data_y = y[i]
        timeIndex = pd.date_range(timeIndex1,timeIndex1+interval-duration, periods = sequenceNum)
        df = pd.DataFrame(index = timeIndex)
        for j, data_columns in enumerate(data_x):   
            df['col_'+str(j)] = data_columns
        df_y.loc[timeIndex[0], 'value'] = data_y
        df_X =pd.concat([df_X, df])

    return df_X, df_y

def trans3NPtoDFbyInputFreq(array, startTime, freq):
    startTimebyData = startTime
    data = pd.DataFrame()
    for array2D_data in array:
        df = trans2NPtoDF(array2D_data, startTimebyData, freq)
        startTimebyData = df.index[-1] + pd.Timedelta(freq)
        
        data = pd.concat([data, df])
        
    return data

def trans2NPtoDF(array, startTime, data_freq):
    """
    Transform 2D Array to DataFrame with indexes and columns transposed.
    
    - array.shape (featureNum, sequenceNum )

    :param array: 2D Array
    :type: 2D numpy array
    
    :param startTime: start Time of DataFrame Index
    :type: string
    
    :param data_freq: time frequency of DataFrame
    :type: string

    >>> startTime : "2022-01-01"
    >>> data_freq : "1S"


    :return df: DataFrame with time stamp as index
    :rtype: DataFrame
    
    """
    seq_len = array.shape[1]
    data_trans = pd.DataFrame(array)
    df = data_trans.T
    timeIndex = pd.date_range(start=startTime, freq = data_freq, periods=seq_len)
    df['datetime'] = timeIndex
    df.set_index(['datetime'], inplace = True)
    
    return df