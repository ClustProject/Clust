import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from Clust.clust.transformation.splitDataByCycle.cycleModule import CycleData

def get_timestep_feature(data, timestep_criteria = {"step":[0, 6, 12, 17, 20, 24], "label":["dawn", "morning", "afternoon", "evening", "night"]}):
    """
    설정된 Time Step에 따라 구분된 Time Label 정보를 "TimeStep" column 에 추가하는 함수
    
    - Hour의 흐름에 따라 구분을 하는 함수로 데이터 시간 정보의 주기가 Hour, Minute, Second 일때 사용

    Returns:
        Time Step 에 따른 Time Label 정보를 포함한 데이터
    """
    
    timestep = timestep_criteria["step"]
    timelabel = timestep_criteria["label"]
    
    data["TimeStep"] = np.array(None)
    for n in range(len(timestep)-1):
        data.loc[data[(data.index.hour >= timestep[n])&(data.index.hour < timestep[n+1])].index, "TimeStep"] = timelabel[n]
    return data

def get_timestepCycleSet_from_dataframe(data, timestep = {"step":[0, 6, 12, 17, 20, 24], "label":["dawn", "morning", "afternoon", "evening", "night"]}):
    # Get data with timestep feature
    data = get_timestep_feature(data, timestep)
    
    # Get Split Cycle Data By Day
    cycle_dataset_by_day = CycleData().getDayCycleSet(data,1,False)
    
    # Get Split timestep Dataset
    timestep_cycle_set = {}
    timestep_label_list = timestep["label"]
    for label in timestep_label_list:
        timestep_cycle_set[label] = []
    
    for cycle_data in cycle_dataset_by_day:
        for label in timestep_label_list:
            split_data = cycle_data[cycle_data["TimeStep"] == label].drop(["TimeStep"], axis=1)
            timestep_cycle_set[label].append(split_data)

    return timestep_cycle_set

def get_timestepCycleSet_from_dataset(dataset, feature_list, timestep):
    timestep_cycle_set_from_dataset = {}
    for feature in feature_list:
        feature_dataset = dataset[feature]
        split_data_list = []
        for data in feature_dataset:
            split_data_list.append(get_timestepCycleSet_from_dataframe(data, timestep))
        timestep_cycle_set_from_dataset[feature] = split_data_list
    
    return timestep_cycle_set_from_dataset