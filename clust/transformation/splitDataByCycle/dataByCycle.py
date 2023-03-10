import sys
import os
sys.path.append("../")
sys.path.append("../../")

from Clust.clust.transformation.splitDataByCycle.cycleModule import CycleData
from Clust.clust.preprocessing.dataPreprocessing import DataPreprocessing
import pandas as pd

def getCycleselectDataFrame(query_data, feature_cycle, feature_cycle_times, frequency=None):
    """
    get Cycle Data

    Args:
        query_data (dataframe): query_data
        feature_cycle (string): feature_cycle
        feature_cycle_times (int): feature_cycle_times
        frequency (int): frequency (option)
    
    Returns:
        List: Cycle Data, or None
    """
    refine_param = {'removeDuplication': {'flag': True}, 'staticFrequency': {'flag': True, 'frequency': frequency}}
    output_data = DataPreprocessing().get_refinedData(query_data, refine_param)
    cycleData = CycleData()

    # cycle 주기에 따라 적절한 함수 적용
    if feature_cycle == 'Hour':
        data = cycleData.getHourCycleSet(output_data, feature_cycle_times, False)
    elif feature_cycle == 'Day':
        data = cycleData.getDayCycleSet(output_data, feature_cycle_times, False)
    elif feature_cycle == 'Week':
        data = cycleData.getWeekCycleSet(output_data, feature_cycle_times, False)
    elif feature_cycle == 'Month':
        data = cycleData.getMonthCycleSet(output_data, feature_cycle_times, False)
    elif feature_cycle == 'Year':
        data = cycleData.getYearCycleSet(output_data, feature_cycle_times, False)

    return data

        
def getCycleSelectDataSet(data_input, feature_cycle, feature_cycle_times, frequency=None):
    """
    get Cycle Data Set

    Args:
        data_input (dataframe or dictionary): query_data
        feature_cycle (string): feature_cycle
        feature_cycle_times (int): feature_cycle_times
        frequency (int): frequency (option)
    
    Returns:
        Dictionary: Cycle DataSet, or None
    """
    
    def _getCycleSelectDataSet_oneDF(data, feature_cycle, feature_cycle_times, frequency):
        """
        get Cycle Data Set

        Args:
            data (dataframe ): query_data
            feature_cycle (string): feature_cycle
            feature_cycle_times (int): feature_cycle_times
            frequency (int): frequency (option)
        
        Returns:
            Dictionary: Cycle DataSet, or None
        """
        data_list = getCycleselectDataFrame(data, feature_cycle, feature_cycle_times, frequency)
        if data_list:
            dataSet = {}
            for data_slice in data_list:
                index_name = data_slice.index[0].strftime('%Y-%m-%d %H:%M:%S')
                dataSet[index_name] = data_slice
        else:
            dataSet = None

        return dataSet

    if isinstance(data_input, dict):
        # if data is dictionary
        result={}
        for data_name in data_input:
            data = data_input[data_name]
            split_data = _getCycleSelectDataSet_oneDF(data, feature_cycle, feature_cycle_times, frequency)
            if split_data:
                result.update(dict((data_name+"/"+key, value) for key, value in split_data.items()))
        
    elif isinstance(data_input, pd.DataFrame):
        # if data is dataframe
        result = _getCycleSelectDataSet_oneDF(data_input, feature_cycle, feature_cycle_times, frequency)

    return result

