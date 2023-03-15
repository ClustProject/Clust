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
        data (dataframe ): query_data
        feature_cycle (string): feature_cycle
        feature_cycle_times (int): feature_cycle_times
        frequency (int): frequency (option)
    
    Returns:
        Dictionary: Cycle DataSet, or None
    """
    result = {}
    data_set = getCycleselectDataFrame(data_input, feature_cycle, feature_cycle_times, frequency)
    if data_set:
        for data in data_set:
            index_name = data.index[0].strftime('%Y-%m-%d %H:%M:%S')
            result[index_name] = data

    return result

