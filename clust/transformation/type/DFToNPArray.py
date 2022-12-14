from decimal import ROUND_UP


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