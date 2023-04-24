
import numpy as np

def trans_DF_to_NP_by_windowNum(df, window_size):
    """_summary_

    Args:
        df (dataframe): input dataframe
        window_size (int): size for slicing

    Returns:
        array (numpy.array): shape ===> (len(df)/window_size , window_size, column_num) ==> (batch, seq_length, input_size )
    """
    
    # 144개 인덱스 간격으로 데이터프레임을 쪼갬
    splitted_df = [df.iloc[i:i+window_size] for i in range(0, len(df), window_size)]
    new_array=[]
    for splitted in splitted_df:
        #list_cols = [splitted[col_name].tolist() for col_name in splitted.columns]
        list_cols = splitted.values.tolist()
        new_array.append(list_cols)
    new_array= np.array(new_array)
    print(new_array.shape)
    
    return new_array
    
    
#YK
def transDFtoNP(dfX, windowNum = 0, dim = None):
    """
    Make NumpyArray by input DataFrame.
    if windowNum = 0 ----> slice X by day
    if windowNum = N ----> slice X by windowNum
    
    Example:
        >>> Retunrn 
        ... X.shape (sampleNum, featureNum, sequenceNum )
        ... y.shape (sampleNum, )

    Args:
        dfX (DataFrame): dfX
        dfy (DataFrame): dfy
        windowNum (Interger): windowNum

    Returns:
        numpy array:  X, y
    
    """
    import datetime as dt
    import numpy as np

    if dim == 2:
        X = dfX.to_numpy()
    else:
        X =[]

        if windowNum ==0:
            dateList = dfX.index.map(lambda t: t.date()).unique()
            for startDate in dateList:
                endDate  = dt.datetime.combine(startDate, dt.time(23, 59, 59, 59))
                dfX_partial = dfX[startDate:endDate]
                X_partial = dfX_partial.values
                X.append (X_partial)
        else:
            import math
            roundNum = math.ceil(len(dfX)/windowNum)
            for i in range(roundNum): #This ensures all rows are captured
                dfX_partial = dfX[i*windowNum:(i+1)*windowNum]
                X_partial = dfX_partial.values
                X.append (X_partial)

        X = np.array(X)
    
    return X

#이전에 쓰던 스타일
def transDFtoNP(dfX, dfy, windowNum = 0, dim = None):
    """
    Make NumpyArray by input DataFrame.
    if windowNum = 0 ----> slice X by day
    if windowNum = N ----> slice X by windowNum
    
    Example:
        >>> Retunrn 
        ... X.shape (sampleNum, featureNum, sequenceNum )
        ... y.shape (sampleNum, )

    Args:
        dfX (DataFrame): dfX
        dfy (DataFrame): dfy
        windowNum (Interger): windowNum

    Returns:
        numpy array:  X, y
    
    """
    import datetime as dt
    import numpy as np

    if dim == 2:
        X = dfX.to_numpy()
        y = np.array(dfy.squeeze().tolist())
    else:
        X =[]
        y= []

        if windowNum ==0:
            dateList = dfX.index.map(lambda t: t.date()).unique()
            for startDate in dateList:
                endDate  = dt.datetime.combine(startDate, dt.time(23, 59, 59, 59))
                dfX_partial = dfX[startDate:endDate]
                dfy_partial = dfy[startDate:endDate]
                X_partial = dfX_partial.values.transpose()
                y_partial = dfy_partial.values[0][0]
                X.append (X_partial)
                y.append (y_partial)
        else:
            import math
            roundNum = math.ceil(len(dfX)/windowNum)
            for i in range(roundNum): #This ensures all rows are captured
                dfX_partial = dfX[i*windowNum:(i+1)*windowNum]
                dfy_partial = dfy[i:(i+1)]
                X_partial = dfX_partial.values.transpose()
                y_partial = dfy_partial.values[0][0]
                X.append (X_partial)
                y.append (y_partial)

        X = np.array(X)
        y = np.array(y)
    
    return X, y




def transDFtoNP2(dfX, windowNum = 0, dim = None):
    """
    Make NumpyArray by input DataFrame.
    if windowNum = 0 ----> slice X by day
    if windowNum = N ----> slice X by windowNum
    
    Example:
        >>> Retunrn 
        ... X.shape (sampleNum, featureNum, sequenceNum )
        ... y.shape (sampleNum, )

    Args:
        dfX (DataFrame): dfX
        dfy (DataFrame): dfy
        windowNum (Interger): windowNum

    Returns:
        numpy array:  X, y
    
    """
    import datetime as dt
    import numpy as np

    if dim == 2:
        X = dfX.to_numpy()
    else:
        X =[]
        y= []

        if windowNum ==0:
            dateList = dfX.index.map(lambda t: t.date()).unique()
            for startDate in dateList:
                endDate  = dt.datetime.combine(startDate, dt.time(23, 59, 59, 59))
                dfX_partial = dfX[startDate:endDate]
                X_partial = dfX_partial.values.transpose()
                X.append (X_partial)
        else:
            import math
            roundNum = math.ceil(len(dfX)/windowNum)
            for i in range(roundNum): #This ensures all rows are captured
                dfX_partial = dfX[i*windowNum:(i+1)*windowNum]
                X_partial = dfX_partial.values.transpose()
                X.append (X_partial)

        X = np.array(X)
    
    return X



















def trans_df_to_np(dfX, dfy, windowNum = 0, dim = None):
    """
    Make NumpyArray by input DataFrame.
    if windowNum = 0 ----> slice X by day
    if windowNum = N ----> slice X by windowNum
    
    Example:
        >>> Retunrn 
        ... X.shape (sampleNum, featureNum, sequenceNum )
        ... y.shape (sampleNum, )

    Args:
        dfX (DataFrame): dfX
        dfy (DataFrame): dfy
        windowNum (Interger): windowNum

    Returns:
        numpy array:  X, y
    
    """
    import datetime as dt
    import numpy as np

    if dim == 2:
        X = dfX.to_numpy()
        y = np.array(dfy.squeeze().tolist())
    else:
        X =[]
        y= []

        if windowNum ==0:
            dateList = dfX.index.map(lambda t: t.date()).unique()
            for startDate in dateList:
                endDate  = dt.datetime.combine(startDate, dt.time(23, 59, 59, 59))
                dfX_partial = dfX[startDate:endDate]
                dfy_partial = dfy[startDate:endDate]
                X_partial = dfX_partial.values.transpose()
                y_partial = dfy_partial.values[0][0]
                X.append (X_partial)
                y.append (y_partial)
        else:
            import math
            roundNum = math.ceil(len(dfX)/windowNum)
            for i in range(roundNum): #This ensures all rows are captured
                dfX_partial = dfX[i*windowNum:(i+1)*windowNum]
                dfy_partial = dfy[i:(i+1)]
                X_partial = dfX_partial.values.transpose()
                y_partial = dfy_partial.values[0][0]
                X.append (X_partial)
                y.append (y_partial)

        X = np.array(X)
        y = np.array(y)
    
    return X, y




def trans_df_to_np_inf(dfX, windowNum = 0, dim = None):
    """
    Make NumpyArray by input DataFrame.
    if windowNum = 0 ----> slice X by day
    if windowNum = N ----> slice X by windowNum
    
    Example:
        >>> Retunrn 
        ... X.shape (sampleNum, featureNum, sequenceNum )
        ... y.shape (sampleNum, )

    Args:
        dfX (DataFrame): dfX
        dfy (DataFrame): dfy
        windowNum (Interger): windowNum

    Returns:
        numpy array:  X, y
    
    """
    import datetime as dt
    import numpy as np

    if dim == 2:
        X = dfX.to_numpy()
    else:
        X =[]
        y= []

        if windowNum ==0:
            dateList = dfX.index.map(lambda t: t.date()).unique()
            for startDate in dateList:
                endDate  = dt.datetime.combine(startDate, dt.time(23, 59, 59, 59))
                dfX_partial = dfX[startDate:endDate]
                X_partial = dfX_partial.values.transpose()
                X.append (X_partial)
        else:
            import math
            roundNum = math.ceil(len(dfX)/windowNum)
            for i in range(roundNum): #This ensures all rows are captured
                dfX_partial = dfX[i*windowNum:(i+1)*windowNum]
                X_partial = dfX_partial.values.transpose()
                X.append (X_partial)

        X = np.array(X)
    
    return X