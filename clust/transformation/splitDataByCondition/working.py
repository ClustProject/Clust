
import pandas as pd
from more_itertools import locate

import sys
import os
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
from Clust.clust.transformation.splitDataByCondition import holiday
from Clust.clust.transformation.splitDataByCycle.cycleModule import CycleData

def get_working_feature(data, workingtime_criteria = {'step': [0, 9, 18, 24], 'label': ['notworking', 'working', 'notworking']}):
    """
    설정된 시간에 따라 일하는 시간과 일하지 않는 시간의 정보를 "Working" column에 추가하는 함수

    - 데이터에 휴일 정보가 있다면 해당 Column의 이름은 "HoliDay" 로 지정한 후에 파라미터에 입력해야한다.
    - 데이터에 휴일 정보가 없다면 휴일을 생성하는 make_holiday_column 함수를 활용하여 휴일정보를 생성
    - 설정된 working_start, working_end 범위 외의 시간과 휴일을 일하지 않는 시간으로 정의
    - 데이터 시간 정보의 주기가 1시간 이하일때 사용

    Returns:
        Working Feature 정보를 포함한 데이터
    """
    if "HoliDay" not in data.columns: 
        data = holiday.get_holiday_feature(data)
    
    work_idx_list = list(locate(workingtime_criteria["label"], lambda x: x == "working"))
    working_row_df = pd.DataFrame()
    for work_idx in work_idx_list:
        working_start = workingtime_criteria["step"][work_idx]
        working_end = workingtime_criteria["step"][work_idx+1]
        
        if working_start <= working_end:
            working_row = pd.Series("working", 
                                    index = data.index[(data.index.hour >= working_start) & (data.index.hour <= (working_end-1))])
        else:
            working_row = pd.Series("working",
                                    index = data.index[(data.index.hour >= working_start) | (data.index.hour <= (working_end-1))])
        
        working_row_df = pd.concat([working_row_df,working_row])
    
    data["Working"] = working_row_df
    data["Working"].fillna("notworking", inplace = True)
    
    # 휴일인 경우
    data.loc[data[data.Holiday == "holiday"].index, "Working"] = "notworking"
    
    return data

def get_workingCycleSet_from_dataframe(data, workingtime_criteria = {'step': [0, 9, 18, 24], 'label': ['notworking', 'working', 'notworking']}):
    # Get data with timestep feature
    data = get_working_feature(data, workingtime_criteria)
    
    # Get Split Cycle Data By Day
    cycle_dataset_by_hour = CycleData().getHourCycleSet(data,1,False)
    
    # Get Split WorkingTime Dataset
    working_data_list = []
    notworking_data_list = []
    
    for cycle_data in cycle_dataset_by_hour:
        if "working" in cycle_data["Working"][0]:
            cycle_data = cycle_data.drop(["Day", "Holiday", "Working"], axis=1)
            working_data_list.append(cycle_data)
        else:
            cycle_data = cycle_data.drop(["Day", "Holiday", "Working"], axis=1)
            notworking_data_list.append(cycle_data)
    
    workingtime_cycle_set = {"working":working_data_list, "notworking":notworking_data_list}
    return workingtime_cycle_set
    
def get_workingCycleSet_from_dataset(dataset, feature_list, workingtime_criteria = {'step': [0, 9, 18, 24], 'label': ['notworking', 'working', 'notworking']}):
    workingtime_cycle_set_from_dataset = {}
    for feature in feature_list:
        feature_dataset = dataset[feature]
        split_data_list = []
        for data in feature_dataset:
            split_data_list.append(get_workingCycleSet_from_dataframe(data, workingtime_criteria))
        workingtime_cycle_set_from_dataset[feature] = split_data_list
    
    return workingtime_cycle_set_from_dataset