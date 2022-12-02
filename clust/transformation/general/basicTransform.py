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

    Args:
        dataset (dictionary): input dataset
        ewm_parameter (float): parameter for ewm function

    Returns:
        dataframe: ssDataSet - scale and smoothed dataframeSet
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


def checkNumericColumns(data, checkColumnList=None):
    """
    This function returns data by trnsforming the Numeric type colums specified in "checkColumnList". 
    If checkColumnList is None, all columns are converted to Numeric type.

    Args:
        data (dataFrame): input Data
        checkColumnList (string array or None): db_name

    Returns:
        dataSet: dataSet

    1. CheckColumnList==None : change all columns to numeric type
    2. CheckColumnList has values: change only CheckColumnList to numeric type
    """
    if checkColumnList:
        pass
    
    else:
        checkColumnList = list(data.select_dtypes(include=['object']).columns)

    data[checkColumnList] = data[checkColumnList].apply(pd.to_numeric, errors='coerce')
    
    return data