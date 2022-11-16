import pandas as pd

#### Duration 
def getRobustScaledDF(DF):
    from sklearn.preprocessing import RobustScaler
    transformer = RobustScaler()
    scaledDF = pd.DataFrame(transformer.fit_transform(DF), index = DF.index, columns = DF.columns )
    return scaledDF

def get_corrMatrix(data):
    corr_matrix = data.corr(method='pearson').values.tolist()
    return corr_matrix

def scalingSmoothingDF(dataSet, ewm_parameter):
    """
    Data can be scaled and smoothed by this function.

    :param dataset: input dataset
    :type dataset: dictionary dataFrameSet
    :param ewm_parameter:  parameter for ewm function
    :type ewm_parameter: float

    :returns ssDataSet: scale and smoothed dataframeSet
    :rtype: dataframe
    """
    ssDataSet=[]
    from sklearn.preprocessing import MinMaxScaler
    for i in range(len(dataSet)):
        value = dataSet[i]
        value= value.ewm(com=ewm_parameter).mean()
        scaler = MinMaxScaler()
       # seriesData_SS_series.append(value.reshape(len(value)))
        df = pd.DataFrame(scaler.fit_transform(value), columns=[dataSet[i].columns], index = dataSet[i].index)
        ssDataSet.append(df)
    return ssDataSet

def DFSetToSeries(dataSet):
    seriesData =[]
    for i in range(len(dataSet)):
        value = dataSet[i].values
        seriesData.append(value.reshape(len(value)))
    return seriesData