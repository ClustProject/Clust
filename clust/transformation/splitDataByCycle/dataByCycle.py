import sys
import os
sys.path.append("../")
sys.path.append("../../")
import datetime
#from Clust.clust.transformation.splitDataByCycle.cycleModule import CycleData
from Clust.clust.preprocessing.dataPreprocessing import DataPreprocessing
import pandas as pd

def getCycleSelectDataSet(data_input, feature_cycle, feature_cycle_times, frequency=None):

    """
    get Cycle Data Set

    Args:
        data_input (dataframe ): query_data
        feature_cycle (string): feature_cycle
        feature_cycle_times (int): feature_cycle_times
        frequency (int): frequency (option)
    
    Returns:
        result(Dictionary): Cycle DataSet, or None
    """
    result = {}
    data_set = getCycleselectDataFrame(data_input, feature_cycle, feature_cycle_times, frequency)

    if data_set:
        for data in data_set:
            index_name = data.index[0].strftime('%Y-%m-%d')
            result[index_name] = data
    return result

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
    refine_param = {"remove_duplication": {'flag': True}, "static_frequency": {'flag': True, 'frequency': frequency}}
    output_data = DataPreprocessing().get_refinedData(query_data, refine_param)
    

    # cycle 주기에 따라 적절한 함수 적용
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
    
    # True 관련해서 체크 안할 것인가?
    if feature_cycle.lower() in ['hour', 'h']:
        unit = "H" # 확인 안해봄
        unit_delta_time = datetime.timedelta(hours=feature_cycle_times)
    elif feature_cycle.lower() in ['day', 'd']:
        unit = "D"
        unit_delta_time = datetime.timedelta(days=feature_cycle_times)
    elif feature_cycle.lower() in ['week', 'w']:
        unit = "W"
        unit_delta_time = datetime.timedelta(weeks=feature_cycle_times)
    elif feature_cycle.lower() in ['month', 'm']:
        unit = "M" # TODO 확인 바람 지훈님   
        unit_delta_time = datetime.timedelta(days=30 * feature_cycle_times)
    elif feature_cycle.lower() in ['year', 'y']:
        unit = "A" # TODO 확인 바람 지훈님   
        unit_delta_time = datetime.timedelta(days=365*feature_cycle_times)
        
    data = cycle_data_set(output_data, unit, unit_delta_time, feature_cycle_times, True)
    
    """
    cycleData = CycleData()
    if feature_cycle in ['Hour', 'H']:
        data = cycleData.getHourCycleSet(output_data, feature_cycle_times, False)
    elif feature_cycle in ['Day', 'D']:
        data = cycleData.getDayCycleSet(output_data, feature_cycle_times, False)
    elif feature_cycle in ['Week', 'W']:
        data = cycleData.getWeekCycleSet(output_data, feature_cycle_times, False)
    elif feature_cycle in ['Month', 'M']:
        data = cycleData.getMonthCycleSet(output_data, feature_cycle_times, False)
    elif feature_cycle in ['Year', 'A']:
        data = cycleData.getYearCycleSet(output_data, feature_cycle_times, False)
    """
    return data

def cycle_data_set(data, unit, unit_delta_time, num, FullCycle):
    # 단위의 데이터 셋 리턴
    """
    Split the data by num*unit

    Args:
        data (dataframe): timeseires data
        unit (str): "H", "D", "W", "M", "A" 중 하나로 시간 단위 나타냄
        num (int): data cycle times
        FullCycle (bool): 완벽한 데이터만 살릴 것인지(True), 양쪽의 군더더기 데이터를 살릴 것인지 (False)
    
    Returns:
        split_data_set(List): 분할된 데이터에 대한 list
    """""
    split_data_set = []

    d_frequency = data.index[1]- data.index[0]
    max_num = int( unit_delta_time /  d_frequency)
    index_name = data.index.name
    split_data = data.reset_index().groupby(pd.Grouper(key = index_name, freq=str(num)+unit))

    # TODO 이후에 num, unit을 그냥 unit_delta_time 으로 변경해도 무방할듯. 우선은 놔둠
    for key, values in split_data:
        split_d = split_data.get_group(key)
        split_d_length = len(split_d)
        split_d = split_d.set_index(index_name)
        split_d.index = pd.to_datetime(split_d.index)
        if FullCycle:
            if split_d_length >= max_num:
                split_data_set.append(split_d)  
            else:
                pass
        else:
            split_data_set.append(split_d)  
    return split_data_set
           


