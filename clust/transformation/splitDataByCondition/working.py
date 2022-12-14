
import pandas as pd
from more_itertools import locate

import sys
import os
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
#from Clust.clust.meta.analysisMeta.meanAnalyzer import holiday

class MeanByWorking():
    def __init__(self, data):
        self.data = data
        self.workingtime_criteria = {'step': [0, 9, 18, 24], 'label': ['notWorking', 'working', 'notWorking']}
        self.labels = ['notWorking', 'working']
    
    def set_workingtime_criteria(self, workingtime_criteria):
        """
        기본적으로 설정된 일하는 시간 기준을 변경하는 함수
        Args:
            workingtime_criteria (dictionary)): 
                - "step", "label" 두 개의 key를 갖고 있다.
                - EX) {'step': [0, 9, 18, 24], 'label': ['notWorking', 'working', 'notWorking']} 는 0시-9시, 18시-24시는 "notWroking" 이며 9시-18시는 "working" 이다.
        """
        self.workingtime_criteria = workingtime_criteria

    # Data WorkingTime Create 
    def make_workingtime_column(self):
        """
        설정된 시간에 따라 일하는 시간과 일하지 않는 시간의 정보를 "Working" column에 추가하는 함수

        - 데이터에 휴일 정보가 있다면 해당 Column의 이름은 "HoliDay" 로 지정한 후에 파라미터에 입력해야한다.
        - 데이터에 휴일 정보가 없다면 휴일을 생성하는 make_holiday_column 함수를 활용하여 휴일정보를 생성
        - 설정된 working_start, working_end 범위 외의 시간과 휴일을 일하지 않는 시간으로 정의
        - 데이터 시간 정보의 주기가 1시간 이하일때 사용

        Returns:
            Working Feature 정보를 포함한 데이터
        """
        #if "HoliDay" not in self.data.columns: 
            #self.data = holiday(self.data).make_holiday_column()
        
        work_idx_list = list(locate(self.workingtime_criteria["label"], lambda x: x == "working"))
        working_row_df = pd.DataFrame()
        for work_idx in work_idx_list:
            working_start = self.workingtime_criteria["step"][work_idx]
            working_end = self.workingtime_criteria["step"][work_idx+1]
            
            if working_start <= working_end:
                working_row = pd.Series("working", 
                                        index = self.data.index[(self.data.index.hour >= working_start) & (self.data.index.hour <= (working_end-1))])
            else:
                working_row = pd.Series("working",
                                        index = self.data.index[(self.data.index.hour >= working_start) | (self.data.index.hour <= (working_end-1))])
            
            working_row_df = pd.concat([working_row_df,working_row])
        
        self.data["Working"] = working_row_df
        self.data["Working"].fillna("notWorking", inplace = True)
        
        # 휴일인 경우
        self.data.loc[self.data[self.data.HoliDay == "holiday"].index, "Working"] = "notWorking"
        
        return self.data
    
    # Data WorkingTime Meta Create
    def get_result(self):
        """
        Working Time&Not Working Time 에 따른 데이터의 평균 값을 Meta 로 생성하는 함수

        - 일하는 시간과 일하지 않는 시간에 따른 분석을 위해 make_workingtime_column 함수로 추출한 "Working" column을 활용
        
        Returns:
            Working Time, Not Working Time 의 평균 값을 포함한 Dictionary Meta
        """
        self.data = self.make_workingtime_column()
        meanbyworking_result_dict = self.data.groupby("Working").mean().to_dict()
        #meanbyworking_result_dict = BasicTool.data_none_error_solution(self.labels, meanbyworking_result_dict)
        
        return meanbyworking_result_dict