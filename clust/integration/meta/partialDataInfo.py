
import sys
sys.path.append("../")
sys.path.append("../..")
import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset
from math import gcd
from functools import reduce

class PartialData():
    def __init__(self, partial_data_set, integration_duration):
        """ 
        This function makes several metadata based on multiple functions

        Args:
            partial_data_set (DataFrameCollection): multiple input dataSet

        Returns:
            json: column_meta
        
        Example:

            >>> partial_data_set ={0: pd.DataFrame, 1: pd.DataFrame, 2:pd.DataFrame}
            >>> from clust.integration.meta import partialDataInfo
            >>> partial_data_info = partialDataInfo.PartialData(partial_data_set)

            >>> partial_data_info.column_meta = {
            ...    "overlap_duration": {
            ...     "start_time": Timestamp("2018-01-01 00:00:00"),
            ...     "end_time": Timestamp("2018-01-01 23:55:00"),
            ... },
            ... "column_characteristics": {
            ...     "data0": {
            ...         "column_name": "data0",
            ...         "column_frequency": Timedelta("0 days 00:10:00"),
            ...         "column_type": dtype("int64"),
            ...         "occurence_time": "Continuous"
            ...         "upsampling_method": "mean",
            ...         "downsampling_method": "mean"
            ...     },
            ...     "data1": {
            ...         "column_name": "data1",
            ...         "column_frequency": Timedelta("0 days 00:07:00"),
            ...         "column_type": dtype("int64"),
            ...         "occurence_time": "Continuous"
            ...         "upsampling_method": "mean",
            ...         "downsampling_method": "mean"
            ...     },
            ...     "data2": {
            ...         "column_name": "data2",
            ...         "column_frequency": Timedelta("0 days 00:03:00"),
            ...         "column_type": dtype("int64"),
            ...         "occurence_time": "Continuous",
            ...         "upsampling_method": "mean",
            ...         "downsampling_method": "mean"
            ...     },
            ...   },
            ... }

            >>> partial_data_info.partial_frequency_info = {'frequency_list': [600.0, 420.0, 180.0],
            ... 'min_frequency': 180.0,
            ... 'max_frequency': 600.0,
            ... 'frequency_is_same': False,
            ... 'average_frequency': 400.0,
            ... 'median_frequency': 420.0,
            ... 'GCD': 60.0,
            ... 'GCDs': '60S'}
            ... partial_data_info.integrated_data_type = AllNumeric

        """
        self.freq_check_length= 3
        self.partial_data_set = partial_data_set
        self.column_meta={}
        self.column_meta['overlap_duration'] = self._get_partial_data_set_start_end(integration_duration)
        self.column_meta['column_characteristics'] = self._get_partial_data_freqeuncy_list(self.freq_check_length)
        self.partial_frequency_info = self._get_partial_data_frequency_info()
        self.integrated_data_type = self._get_partial_data_type()
    
    def _get_partial_data_freqeuncy_list(self, freq_check_length):
    
        data_length = len(self.partial_data_set)
        column_list={}
        for i in range(data_length):
            data = self.partial_data_set[i]
            columns = data.columns
            for column in columns:
                column_info = {"column_name":'', "column_frequency":'', "column_type":''}
                column_info['column_name'] = column
                freq = self.get_df_freq_timedelta(data, freq_check_length)
                column_info['column_frequency'] = freq
                if freq is not None:
                    column_info['occurence_time'] = "Continuous"
                else:
                    column_info['occurence_time'] = "Event" 
                column_info['pointDependency']="Yes" #default

                column_type = data[column].dtype
                column_info['column_type'] = column_type
                
                if column_type == np.dtype('O') :
                    column_info['upsampling_method']= "objectUpFunc" 
                    column_info['downsampling_method']= "objectDownFunc"
                else:
                    column_info['upsampling_method']="mean" #default
                    column_info['downsampling_method']="mean" #default
                column_list[column] = column_info
        return column_list
     
    def _get_partial_data_frequency_info(self):
        
        partialFreqList=[]
        data_length = len(self.partial_data_set)
        for i in range(data_length):
            freq = self.get_df_freq_sec(self.partial_data_set[i], self.freq_check_length)
            partialFreqList.append(int(freq))
            
        frequency={}
        frequency['frequency_list']= partialFreqList
        frequency['min_frequency'] = min(partialFreqList)
        frequency['max_frequency'] = max(partialFreqList)
        frequency['frequency_is_same'] = self._check_same_freq(partialFreqList)
        frequency['average_frequency'] = np.mean(partialFreqList)
        frequency['median_frequency'] = np.median(partialFreqList)

        def find_gcd(list):
            x = reduce(gcd, list)
            return x
        frequency_list = frequency['frequency_list']
        frequency['GCD'] = find_gcd(frequency_list) #Greatest common divisor
        frequency['GCDs'] = str(int(frequency['GCD']))+'S'

        return frequency
    
    def _get_partial_data_set_start_end(self, integration_duration):
        start_list=[]
        end_list =[]
        duration={}
        data_length = len(self.partial_data_set)
        for i in range(data_length):
            data =self.partial_data_set[i]
            start = data.index[0]
            end = data.index[-1]
            start_list.append(start)
            end_list.append(end)
        if integration_duration == 'common':
            duration['start_time'] = max(start_list)
            duration['end_time'] = min(end_list)
        elif integration_duration == 'total':
            duration['start_time'] = min(start_list)
            duration['end_time'] = max(end_list)
        
        return duration
    
    # AllNumeric, AllCategory, NumericCategory
    def _get_partial_data_type(self):
        numCols_list =[]
        catCols_list=[]
        data_length = len(self.partial_data_set)
        for i in range(data_length):
            data =self.partial_data_set[i]
            numCols = data.select_dtypes("number").columns
            catCols = data.select_dtypes(include=["object", "bool", "category"]).columns
            numCols_list.extend(numCols.values)
            catCols_list.extend(catCols.values)
        if len(numCols_list) >  0 :
            if len(catCols_list) == 0 :
                datatype = "AllNumeric"
            else:
                datatype ="NumericCategory"
        else:
            datatype = "AllCategory"
          
        return datatype
    
    
    def _check_same_freq(self, item_list):
        return(len(set(item_list))) < 2

    def get_df_freq_sec(self, data, freq_check_length):

        freq = to_offset(pd.infer_freq(data[:freq_check_length].index))
        if not freq:
            freq_sec = (data.index[1] - data.index[0]).total_seconds()
        else:
            freq_sec = pd.to_timedelta(freq, errors='coerce').total_seconds()
        return freq_sec

    def get_df_freq_timedelta(self, data, freq_check_length):
        freq = to_offset(pd.infer_freq(data[:freq_check_length].index))
        if not freq:
            freq_timedelta= pd.to_timedelta(data.index[1] - data.index[0])
        else:
            freq_timedelta = pd.to_timedelta(freq, errors='coerce')
        return freq_timedelta

        
    