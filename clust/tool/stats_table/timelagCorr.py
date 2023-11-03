import math
import numpy as np
import pandas as pd

class TimeLagCorr():
    def __init__(self):
        """
        - Calculate cross correlation with time lag. 
        - This class provides function for dataframe and series data input. 
        
        """
        pass

    
    def df_timelag_crosscorr(self, dataSet, column, lag_number:int):
        """
        Calculate timelag crosscorrelation for dataframe input

        Args:
            dataSet (_pd.dataFrame_) : Input DataSet to be calculated
            column (_str_) : reference one column name
            lag_number (_int_) : max range to investigate time difference (-lag_number ~ lag_number)

        Returns:
            _pd.dataFrame_ : result(cross-correlation dataFrame result with time_lag index)

        >>> result = (index = time_lag, columns = columns, value = cross-correlation values)

        """
        lags    = np.arange(-lag_number, lag_number, 1)
        d1      = dataSet[column]
        d2DF    = dataSet.drop(column, axis=1)

        result = pd.DataFrame(index = lags)
        for d2_column in d2DF.columns:
            d2 = d2DF[d2_column]
            result[d2_column] = self.timelag_crosscorr(d1, d2, lags)

        result = result.round(2)
        
        return result
        
     
    def timelag_crosscorr(self, datax:pd.Series, datay:pd.Series, lags:int) -> pd.Series :
        """
        Calculate timelagged crosscorrelation for series data

        Args:
            datax (_pd.Series_) : input series
            datay (_pd.Series_) : reference series
            lags (_int_) : max range to investigate time difference (-lag_number ~ lag_number)

        Returns:
            np.array : result(cross-correlation series result with time_lag index)

        """
        
        timeLagCrossCorr =[]
        result = np.nan_to_num([self.crosscorr(datax, datay, lag) for lag in lags])

        return result
        
    def crosscorr(self, datax, datay, lag=0):
        """ 
        Lag-N cross correlation. Shifted data filled with NaNs. 

        Args:
            datax, datay (_pd.Series_) : Series objects of equal length
            lag (_int_) : default 0
        
        Returns:
            _float_ : crosscorr
        
        """
        return datax.corr(datay.shift(lag))

    def get_absmax_index_and_values(self, data):
        """
        get dataframe with max cross correlation values and its index.

        Args:
            data (_pd.dataframe_) : data

        Returns:
            _pd.DataFrame_ : max_position_value

        max_position_value :

            >>> index = name of columns
            ... columns = ['value', 'index']
            ... value = 'value': max value, 'index':its index value
        
        """
        
        max_position_value = pd.DataFrame(index = data.columns, columns =['value', 'index'])
        for column in data.columns:
            index_num = data.abs().idxmax()[column]
            max_position_value.loc[column, "index"] = index_num
            max_position_value.loc[column, "value"] = data.loc [index_num, column]

        return max_position_value