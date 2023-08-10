
import numpy as np


from Clust.clust.transformation.purpose import machineLearning as ML
def trans_DF_to_NP_by_windowNum(X, y, transformParameter):
    """_summary_

    Args:
        df (dataframe): input dataframe
        window_size (int): size for slicing

    Returns:
        array (numpy.array): shape ===> (len(df)/window_size , window_size, column_num) ==> (batch, seq_length, input_size )
    """
    window_size = transformParameter['past_step']
    max_nan_limit_ratio = transformParameter['max_nan_limit_ratio']
    nan_limit_num = int(window_size*max_nan_limit_ratio)
    print("window_size:", window_size, "nan_limit_num:", nan_limit_num)
    y_array_old = y.values
    # 144개 인덱스 간격으로 데이터프레임을 쪼갬
    splitted_df = [X.iloc[i:i+window_size] for i in range(0, len(X), window_size)]
    X_array, y_array=[], []
    print(nan_limit_num, "만큼의 에러가 있는 데이터는 생략함")
    for i, splitted in enumerate(splitted_df):
        #list_cols = [splitted[col_name].tolist() for col_name in splitted.columns]
        np_X = np.array(splitted.values.tolist())
        np_y = np.array(y_array_old[i])
        np_X , np_y = ML.check_nan_status(np_X, np_y, nan_limit_num)
        
        # step2
        if np.isnan(np_X).any() | np.isnan(np_y).any():
            pass
        else:
            X_array.append(np_X)
            y_array.append(np_y)
    X_array = np.array(X_array)
    y_array = np.array(y_array)
    print(X.shape, X_array.shape)
    print(y.shape, y_array.shape)
    
    return X_array, y_array
    
    
#YK
def transDFtoNP_infer(dfX, windowNum = 0, dim = None):
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