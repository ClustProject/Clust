
import pandas as pd
from more_itertools import locate

import sys
import os
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
from Clust.clust.transformation.general.basicTransform import nan_to_none_in_dict
from Clust.clust.transformation.splitDataByCondition import working

def get_mean_analysis_result_by_workingtime(data, workingtime_criteria = {'step': [0, 9, 18, 24], 'label': ['notworking', 'working', 'notworking']}):
    """
    Working Time&Not Working Time 에 따른 데이터의 평균 값을 Meta 로 생성하는 함수
    일하는 시간과 일하지 않는 시간에 따른 분석을 위해 make_workingtime_column 함수로 추출한 "Working" column을 활용
    
    Returns:
        Dictionary : Working Time, Not Working Time 의 평균 값을 포함한 Dictionary Meta
    """
    labels = ['notworking', 'working']
    data = working.get_working_feature(data, workingtime_criteria)
    meanbyworking_result_dict = data.groupby("Working").mean().to_dict()
    meanbyworking_result_dict = nan_to_none_in_dict(labels, meanbyworking_result_dict)
    
    return meanbyworking_result_dict