from unittest import result
from pytimekr import pytimekr
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from Clust.clust.meta.analysisMeta.basicTool import BasicTool

class MeanByHoliday():
    def __init__(self, data):
        self.data = data
        self.holiday_criteria = {"step":["Sat", "Sun"], "label":["Holiday"]}
        self.labels = ["holiday", "notHoliday"]
        
    def set_holiday_criteria(self, holiday_criteria):
        """
        기본적으로 설정된 휴일의 기준을 변경하는 함수
        
        Args:
            timestep_criteria (dictionary)): 
                - "step", "label" 두 개의 key를 갖고 있다.
                - EX) {"step":["Sat", "Sun"], "label":["Holiday"]}은 "Sat", "Sun"이 "Holiday"라는 의미이다.
        """
        self.holiday_criteria = holiday_criteria

    def make_day_column(self): # make_day_info
        """
        데이터의 시간에 따라 "Day" column에 요일 정보를 추가하는 함수

        Returns:
            요일 정보를 포함한 데이터
        """
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_list = [days[x] for x in self.data.index.weekday]
        self.data["Day"] = day_list
        return self.data
    
    # Data Public Holiday Create
    def make_holiday_column(self): # make_holiday_info
        """
        데이터의 시간에 따라 "HoliDay" column에 휴일 정보를 추가하는 함수
        요일 정보를 생성해주는 함수를 활용해서 주말 정보를 휴일로 구분
        휴일은 설정된 휴일 기준으로 한다.

        Returns: 
            요일, 휴일 정보를 포함한 데이터
        """
        self.data = self.make_day_column()
        
        ## public_holiday 생성
        years = range(self.data.index.min().year, self.data.index.max().year+1)
        public_holiday_list = []
        for year in years:
            public_holiday_list += [x.strftime("%Y-%m-%d") for x in pytimekr.holidays(year)]
        
        ## 주말 list 생성
        
        input_holiday_list = []
        for input_holiday in self.holiday_criteria["step"]:
           input_holiday_list += [x.strftime("%Y-%m-%d") for x in self.data[self.data.Day == input_holiday].index.tz_convert(None)]
        input_holiday_set = set(input_holiday_list)
        
        ## 최종 휴일 Column 생성 (공휴일 + holiday_criteria 기준)
        final_holiday_list = public_holiday_list + list(input_holiday_set)
        holidays = ["holiday" if x.strftime("%Y-%m-%d") in final_holiday_list else "notHoliday" for x in self.data.index]
        self.data["HoliDay"] = holidays

        return self.data
    
    def get_result(self):
        """
        Holiday &Not Holiday 에 따른 데이터의 평균 값을 Meta 로 생성하는 함수

        - 휴일과 휴일이 아닌 날에 따른 분석을 위해 make_holiday_column 함수로 추출한 "HoliDay" column을 활용
        
        Returns:
            Holiday, Not Holiday 의 평균 값을 포함한 Dictionary Meta
        """
        self.data = self.make_holiday_column()
        print(">>>>> make holiday column success <<<<<")
        meanbyholiday_result_dict = self.data.groupby("Holiday").mean().to_dict()
        print(">>>>> holiday groupby success <<<<<")
        meanbyholiday_result_dict = BasicTool.data_none_error_solution(self.labels, meanbyholiday_result_dict)
        
        return meanbyholiday_result_dict