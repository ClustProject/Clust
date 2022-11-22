import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from KETIToolAnalyzer.basicTool import BasicTool

class MeanByTimeStep():
    def __init__(self, data):
        self.data = data
        self.timestep_criteria = {"step":[0, 6, 12, 17, 20, 24], "label":["dawn", "morning", "afternoon", "evening", "night"]}
        
    def set_timestep_criteria(self, timestep_criteria):
        """
        기본적으로 설정된 timestep의 기준을 변경하는 함수
        Args:
            timestep_criteria (dictionary)): 
                - "step", "label" 두 개의 key를 갖고 있다.
                - EX) {"step":[0, 6, 12, 17, 20, 24], "label":["dawn", "morning", "afternoon", "evening", "night"]} 에서 0시-6시는 "dawn" 을 뜻한다.
        """
        self.timestep_criteria = timestep_criteria
        
    def make_timestep_column(self):
        """
        설정된 Time Step에 따라 구분된 Time Label 정보를 "TimeStep" column 에 추가하는 함수
        
        - Hour의 흐름에 따라 구분을 하는 함수로 데이터 시간 정보의 주기가 Hour, Minute, Second 일때 사용

        Returns:
            Time Step 에 따른 Time Label 정보를 포함한 데이터
        """
        
        timestep = self.timestep_criteria["step"]
        timelabel = self.timestep_criteria["label"]
        
        self.data["TimeStep"] = np.array(None)
        for n in range(len(timestep)-1):
            self.data.loc[self.data[(self.data.index.hour >= timestep[n])&(self.data.index.hour < timestep[n+1])].index, "TimeStep"] = timelabel[n]
        return self.data

    # Data Time Step Meta Create 
    def get_result(self):
        """
        설정된 Time Step, Time Label 에 따른 데이터의 평균 값을 Meta 로 생성하는 함수

        - Time Step 에 따른 Time Label 정보를 분석하기 위해 make_timestep_column 함수로 추출한 "TimeStep" column 을 활용

        Returns:
            Time Label 에 따른 평균 값을 포함한 Dictionary Meta
        """
        self.data = self.make_timestep_column()
        meanbytimestep_result_dict = self.data.groupby("TimeStep").mean().to_dict()
        meanbytimestep_result_dict = BasicTool.data_none_error_solution(self.timestep_criteria["label"], meanbytimestep_result_dict)
        
        return meanbytimestep_result_dict
