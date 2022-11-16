import sys
import os
sys.path.append("../")
sys.path.append("../../")

from clust.quality.cycle.cycleData import CycleData
from clust.preprocessing.data_preprocessing import DataPreprocessing


def getCycleselectDataFrame(query_data, feature_cycle, feature_cycle_times, frequency=None):
    """
    get Cycle Data

    :param query_data: query_data
    :type query_data: dataframe

    :param feature_cycle: feature_cycle
    :type feature_cycle: string

    :param feature_cycle_times:feature_cycle_times
    :type feature_cycle_times: int

    :param frequency: frequency
    :type frequency: int
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


def getCycleSelectDataSet(query_data, feature_cycle, feature_cycle_times, frequency=None):
    """
    get Cycle Data Set

    :param query_data: query_data
    :type query_data: dataframe

    :param feature_cycle: feature_cycle
    :type feature_cycle: string

    :param feature_cycle_times:feature_cycle_times
    :type feature_cycle_times: int

    :param frequency: frequency
    :type frequency: int
    """
    data_list = getCycleselectDataFrame(query_data, feature_cycle, feature_cycle_times, frequency)
    dataSet = {}
    for data_slice in data_list:
        index_name = data_slice.index[0].strftime('%Y-%m-%d %H:%M:%S')
        dataSet[index_name] = data_slice

    return dataSet