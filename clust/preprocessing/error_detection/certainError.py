
import numpy as np
import pandas as pd

class CertainErrorRemove():
    '''Let Certain Outlier from DataFrame Data to NaN. This function makes more Nan according to the data status.
    
    **Data Preprocessing Modules**::

            ``Sensor Min Max Check``, ``Remove no numeric data``
    '''
    def __init__(self, data, min_max_limit, anomal_value_list=[99.9, 199.9, 299.9, 9999, -99.9, -199.9, -299.9, -9999, -9999.0] ):
        # TODO JW min_max 통과하는 모듈도 업그레이드 해야함
        self.anomal_value_list = anomal_value_list
        self.data = data
        self.min_max_limit = min_max_limit
    
    def getDataWitoutcertainError(self):
        #Main Function
        # - Delete duplicated data
        # - Delete Out of range error 

        data_out = self.data.copy()
        data_out = self._out_of_range_error_remove (data_out, self.min_max_limit)
        # TODO JW anomal_value_list 관련 향후 수정/업그레이드 해야 함 
        data_out = self._anomal_value_remove(data_out, self.anomal_value_list)
        return data_out
        
    def _out_of_range_error_remove (self, data, min_max_limit):
        """ Remove out-of-range errors and outliers. change error values to NaN

        :param data: input data
        :type data: DataFrame 
        :param min_max_limit: min_max_limit information
        :type min_max_limit: json

        :return: New Dataframe having more (or same) NaN
        :rtype: DataFrame

        **Two Outlier Detection Modules**::
            - remove _out_of_range_error
            - Delete Out of range error
        
        example
            >>> output = CertainErrorRemove().getDataWitoutcertainError(daata, min_max_limit)     
        """

        data_out = data.copy()
        column_list = data.columns
        max_list = min_max_limit['max_num']
        min_list = min_max_limit['min_num']
        
        for column_name in column_list:
            if column_name in min_list.keys():
                min_num = min_list[column_name]
                mask = data_out[column_name] < min_num
                #merged_result.loc[mask, column_name] = min_num
                data_out[column_name][mask] = np.nan #min_num

            if column_name in max_list.keys():
                max_num = max_list[column_name]
                mask = data_out[column_name] > max_num
                data_out[column_name][mask] = np.nan #max_num

        return data_out

    def _anomal_value_remove(self, data, anomal_value_list):
        """ Remove out-of-range errors. change error values to NaN

        :param data: input data
        :type data: DataFrame 
        :param min_max_limit: min_max_limit information
        :type min_max_limit: json
        
        :return: New Dataframe having more (or same) NaN
        :rtype: DataFrame
        
        example
            >>> output = CertainErrorRemove().remove_out_of_range_error(data, min_max_limit)     
        """

        # 특정 이상치 nan 처리 
        anomal_data = anomal_value_list 
        for index in anomal_data:
            data = data.replace(index, np.NaN)
        return data
