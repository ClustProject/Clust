import pandas as pd
from more_itertools import locate

import sys
import os
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
from Clust.clust.transformation.splitDataByCondition import holiday

def get_working_feature(data, workingtime_criteria = {'step': [0, 9, 18, 24], 'label': ['notworking', 'working', 'notworking']}):
    """
    A function that adds a "Working" column constructed according to the input working time.

    - If there is holiday information in the data, the name of the corresponding column must be set as "HoliDay" and then entered in the parameter. 
      If there is no holiday information in the data, the function automatically creates it.
    - Since the function is classified based on Hour, the Input data time frequency must be Hour, Minute, or Second.

    Args:
        data (dataframe) : Time series data
        workingtime_criteria (dictionary) : Working time criteria information
        
    Example:
        >>> workingtime_criteria = {'step': [0, 9, 18, 24], 'label': ['notworking', 'working', 'notworking']}
    
    Returns:
        dataframe : Time sereis data with "Working" column    
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

def get_split_data_by_workingtime_from_dataframe(data, workingtime_criteria = {'step': [0, 9, 18, 24], 'label': ['notworking', 'working', 'notworking']}):
    """
    Split the data by working/notworking.

    Args:
        data (dataframe): Time series data
        workingtime_criteria (dictionary) : Working time criteria information
        
    Example:
        >>> workingtime_criteria = {'step': [0, 9, 18, 24], 'label': ['notworking', 'working', 'notworking']}

    Returns:
        dictionary: Return value composed of dataframes divided according to each label of working and notworking.
    """
    # Get data with working feature
    data = get_working_feature(data, workingtime_criteria)
    
    # Split Data by Working time
    split_data_by_working = {}
    split_data_by_working["working"] = data["working" == data["Working"]].drop(["Day", "Holiday", "Working"], axis=1)
    split_data_by_working["notworking"] = data["notworking" == data["Working"]].drop(["Day", "Holiday", "Working"], axis=1)
    
    return split_data_by_working
    
def get_split_data_by_workingtime_from_dataset(dataset, workingtime_criteria = {'step': [0, 9, 18, 24], 'label': ['notworking', 'working', 'notworking']}):
    """
    Split Data Set by working/notworking.

    Args:
        dataset (Dictionary): dataSet, dictionary of dataframe (ms data). A dataset has measurements as keys and dataframes(Timeseries data) as values.
        workingtime_criteria (dictionary) : Working time criteria information
        
    Example:
        >>> workingtime_criteria = {'step': [0, 9, 18, 24], 'label': ['notworking', 'working', 'notworking']}
    
    Returns:
        dictionary: Return value has measurements as keys and split result as values. 
                    split result composed of dataframes divided according to each label of working and notworking.
    """
    split_dataset_by_working = {}
    for ms_name in dataset:
        data = dataset[ms_name]
        if not(data.empty):
            split_data_by_working_dict = get_split_data_by_workingtime_from_dataframe(data, workingtime_criteria)
            split_dataset_by_working[ms_name] = split_data_by_working_dict

    return split_dataset_by_working