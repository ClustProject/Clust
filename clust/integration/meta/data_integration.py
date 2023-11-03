import pandas as pd 
import numpy as np
from collections import Counter
import math 
class DataIntegration():
    """
    Data Integration Class
    """
    def __init__(self, data_partial):    
        self.data_partial = data_partial
        
    def dataIntegrationByMeta(self, re_frequency, column_meta):
        """ 
        1. simple data integration
        2. restructure data with new frequency
        3. fill NA resulting from integration

        Args:
            re_frequency (json): metadata for integration of each dataset 
            column_characteristics (json): metadata for integration of each dataset  
            
        Returns:
            DataFrame: integrated_data_resample_fillna    
        """
        column_characteristics = column_meta['column_characteristics']
        self.simple_integration(column_meta['overlap_duration'])
        
        integrated_data_resample = self.restructured_data_with_new_frequency(re_frequency, column_characteristics)
        integrated_data_resample_fillna = self.restructured_data_fillna(integrated_data_resample, column_characteristics,re_frequency )

        return integrated_data_resample_fillna

    def simple_integration(self, duration):
        """ 
        This function integrates datasetCollection without no pre-post processing.
        
        Args:
            duration (dictionary): duration
            
        Returns:
            DataFrame: merged_data

        Example:
            >>> duration: {'start_time': Timestamp('2018-01-01 00:00:00'), 
            ...             'end_time': Timestamp('2018-01-01 23:55:00')}

        >>> from clust.integration.meta import data_integration
        >>> data_int = data_integration.DataIntegration(data_partial_numeric)
        >>> integrated_data = data_int.simple_integration(partial_data_info.column_meta['overlap_duration'])

        """
        data_key_list = list(self.data_partial.keys())
        merged_data_list =[]
        for data_name in data_key_list:
            partial_data = self.data_partial[data_name].sort_index(axis=1)
            merged_data_list.append(partial_data)
        merged_data = pd.concat(merged_data_list, axis=1, join='outer', sort=True)#inner

        start_time = duration['start_time']
        end_time = duration['end_time']
        merged_data = merged_data[start_time:end_time]
        self.merged_data = merged_data 
        return self.merged_data
    
        
    def restructured_data_with_new_frequency(self, re_frequency, column_characteristics):
        """ This function integrates datasetCollection with new data frequency
        
        Args:
            re_frequncy (timedelta): description frequency for new integrated data
            column_characteristics (json): metadata for integration of each dataset

        >>> column_characteristics = {
        ...          "data0": {
        ...             "column_name": "data0",
        ...             "column_frequency": Timedelta("0 days 00:10:00"),
        ...             "column_type": dtype("int64"),
        ...             "occurence_time": "Continuous",
        ...             "pointDependency": "Yes",
        ...             "upsampling_method": "mean",
        ...             "downsampling_method": "mean",
        ...             "fillna_function": "interpolate",
        ...             "fillna_limit": 31,
        ...         },
        ...         "data1": {
        ...             "column_name": "data1",
        ...             "column_frequency": Timedelta("0 days 00:07:00"),
        ...             "column_type": dtype("int64"),
        ...             "occurence_time": "Continuous",
        ...             "pointDependency": "Yes",
        ...             "upsampling_method": "mean",
        ...             "downsampling_method": "mean",
        ...             "fillna_function": "interpolate",
        ...             "fillna_limit": 31,
        ...         },
        ...         "data2": {
        ...             "column_frequency": Timedelta("0 days 00:03:00"),
        ...             "column_type": dtype("int64"),
        ...             "occurence_time": "Continuous",
        ...             "pointDependency": "Yes",
        ...             "upsampling_method": "mean",
        ...             "downsampling_method": "mean",
        ...             "fillna_function": "interpolate",
        ...             "fillna_limit": 31,
        ...         },
        ... }

        Returns:
            DataFrame: merged_data

        >>> from clust.integration.meta import data_integration
        >>> data_int = data_integration.DataIntegration(dataset)
        >>> re_frequency = datetime.timedelta(seconds=180)
        >>> integrated_data_resample = data_int.restructured_data_with_new_frequency(re_frequency, column_characteristics)

        """
        
        # TODO JW 수정해서 바꿔야함. 우선 스트링 타입에 대해서 이젠 동작하지 않고 있음
        column_function={}
        for column_name in column_characteristics:
            #reStructuredData = data.resample(frequency).apply(np.mean)
            column_info = column_characteristics[column_name]
            origin_frequency = column_info['column_frequency']
            if origin_frequency <= re_frequency: #down_sampling
                sampling_method_string = column_info['downsampling_method']
            if origin_frequency > re_frequency: #upsampling
                sampling_method_string = column_info['upsampling_method']
            sampling_method = self.converting_sampling_method(sampling_method_string)
            column_function[column_name] = sampling_method

        # To Do : Upgrade merge function 
        reStructuredData = self.merged_data.resample(re_frequency).agg(column_function)  
        return reStructuredData 

    def converting_sampling_method(self, sampling_method_string):
        """
        Description 추가 필요

        Args:
            sampling_method_string(string) : mean or median or objectDownFunc or objectUpFunc

        Resturns:
            np.array : sampling_method
        """
        def objectDownFunc(x):
            c = Counter(x)
            mostFrequent = c.most_common(1)
            return mostFrequent[0][0]

        def objectUpFunc(x): # TODO Modify
            c = Counter(x)
            mostFrequent = c.most_common(1)
            if 0< len(mostFrequent):
                result = mostFrequent[0][0]
            else:
                result =np.NaN
            return result

        if sampling_method_string =='mean':
            sampling_method = np.mean
        elif sampling_method_string =='median':
            sampling_method = np.median
        elif sampling_method_string == 'objectDownFunc':
            sampling_method=objectDownFunc
        elif sampling_method_string =='objectUpFunc':
            sampling_method = objectUpFunc

        return sampling_method
        
    def restructured_data_fillna(self, origin_data, column_characteristics,re_frequency):
        """ This function integrates datasetCollection and fill NA
        
        Args:
            origin_data (DataFrame): integrated with resampling NaN
            column_characteristics (json): metadata for integration of each dataset  
            re_frequency (json): metadata for integration of each dataset 
            
        Returns:
            DataFrame: reconstructedData

        >>> from clust.integration.meta import data_integration
        >>> data_int = data_integration.DataIntegration(dataset)
        >>> re_frequency = datetime.timedelta(seconds=180)
        >>> integrated_data_resample = data_int.restructured_data_with_new_frequency(re_frequency, column_characteristics)
        >>> integrated_data_resample_fillna = data_int.restructured_data_fillna(integrated_data_resample, column_characteristics,re_frequency, fillna_num )
        """
        column_function={} 
        
        reStructuredData = origin_data.copy()
        for column_name in column_characteristics:
            #reStructuredData = data.resample(frequency).apply(np.mean)
            column_info = column_characteristics[column_name]
            origin_frequency = column_info['column_frequency']
            limit_num = math.ceil(origin_frequency/re_frequency)
            if origin_frequency > re_frequency: #upsampling
                if column_info['column_type'] == np.dtype('O'):
                   reStructuredData[column_name] = reStructuredData[column_name].fillna(method ="ffill", limit = limit_num) 
                else:
                    reStructuredData[column_name] = reStructuredData[column_name].interpolate(limit = limit_num) 

        return reStructuredData 
    


