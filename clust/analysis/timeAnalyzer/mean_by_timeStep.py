import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from Clust.clust.transformation.general.basicTransform import nan_to_none_in_dict
from Clust.clust.transformation.splitDataByCondition import timeStep

def get_mean_analysis_result_by_timestep(data, timestep_criteria = {"step":[0, 6, 12, 17, 20, 24], "label":["dawn", "morning", "afternoon", "evening", "night"]}):
    """
    설정된 Time Step, Time Label 에 따른 데이터의 평균 값을 Meta 로 생성하는 함수

    - Time Step 에 따른 Time Label 정보를 분석하기 위해 make_timestep_column 함수로 추출한 "TimeStep" column 을 활용

    Returns:
        Time Label 에 따른 평균 값을 포함한 Dictionary Meta
    """
    data = timeStep.get_timestep_feature(data, timestep_criteria)
    meanbytimestep_result_dict = data.groupby("TimeStep").mean().to_dict()
    meanbytimestep_result_dict = nan_to_none_in_dict(timestep_criteria["label"], meanbytimestep_result_dict)
    
    return meanbytimestep_result_dict