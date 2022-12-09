from unittest import result
from pytimekr import pytimekr
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from Clust.clust.meta.analysisMeta.basicTool import BasicTool
from Clust.clust.transformation.splitDataByCycle.cycleModule import CycleData

def get_holiday_feature(data):
    """
    A function that adds a new holiday column by designating holidays and weekends as holiday days during the data period.

    Args:
        data (dataframe) : Time series data

    Returns:
        dataframe : Time sereis data with "Holiday" column
    """
    # 요일 Feature 생성
    data["Day"] = data.index.day_name()

    ## 주말 list 생성
    week_list = list(set(data[(data.Day == "Saturday") | (data.Day == "Sunday")].index.strftime("%Y-%m-%d")))
    
    ## public_holiday list 생성
    years = range(data.index.min().year, data.index.max().year+1)
    public_holiday_list = []
    for year in years:
        public_holiday_list += [x.strftime("%Y-%m-%d") for x in pytimekr.holidays(year)]
    
    ## 최종 휴일 Feature 생성 (공휴일 + sunday/saturay 기준)
    final_holiday_list = public_holiday_list + week_list
    holidays_not_holiday_list = ["holiday" if x.strftime("%Y-%m-%d") in final_holiday_list else "notholiday" for x in data.index]
    data["Holiday"] = holidays_not_holiday_list

    return data

def get_holiday_cycle_set_by_dataframe(data):
    """
    Split the data by holiday/non-holiday.

    Args:
        data (dataframe): Time series data

    Returns:
        _type_: _description_
    """
    # Get data with holiday feature
    data = get_holiday_feature(data)
    
    # Get Split Cycle Data By Day
    cycle_dataset_by_day = CycleData().getDayCycleSet(data,1,False)
    
    # Get Split holiday&notholiday Dataset
    holiday_data_list = []
    notholiday_data_list = []
    for cycle_data in cycle_dataset_by_day:
        if "holiday" in cycle_data["HoliDay"][0]:
            holiday_data_list.append(cycle_data)
        else:
            notholiday_data_list.append(cycle_data)
    
    holiday_cycle_set = {"holiday":holiday_data_list, "notholiday":notholiday_data_list}
    return holiday_cycle_set

def get_holiday_cycle_set_by_dataset(dataset):
    # feature_list = list(dataset.keys())
    # for feature_name in feature_list:
    #     feature_dataset = dataset[feature_name]
    #     for data in feature_dataset:
    #         get_holiday_cycle_set_by_dataframe
    pass


# class SplitDataByHoliday():
#     """
#     휴일을 기준으로 데이터를 나누는 모듈
    
#     Args:
#         data (dataframe) : Time Series Data
#     """
#     def __init__(self, data):
#         self.data = data
        
#     def get_holiday_feature(self):
#         """
#         해당 데이터 시간 구간의 공휴일, 주말에 관한 휴일 컬럼을 새롭게 추가하는 함수

#         Returns:
#             dataframe : Time sereis data with "Holiday" column
#         """
#         # 요일 Feature 생성
#         self.data["Day"] = self.data.index.day_name()

#         ## 주말 list 생성
#         week_list = list(set(self.data[(self.data.Day == "Saturday") | (self.data.Day == "Sunday")].index.strftime("%Y-%m-%d")))
        
#         ## public_holiday list 생성
#         years = range(self.data.index.min().year, self.data.index.max().year+1)
#         public_holiday_list = []
#         for year in years:
#             public_holiday_list += [x.strftime("%Y-%m-%d") for x in pytimekr.holidays(year)]
        
#         ## 최종 휴일 Feature 생성 (공휴일 + sunday/saturay 기준)
#         final_holiday_list = public_holiday_list + week_list
#         holidays_not_holiday_list = ["holiday" if x.strftime("%Y-%m-%d") in final_holiday_list else "notholiday" for x in self.data.index]
#         self.data["Holiday"] = holidays_not_holiday_list

#         return self.data


#     def get_holiday_cycle_set(self):
#         pass
    
    
    
#     def set_holiday_criteria(self, holiday_criteria):
#         """
#         기본적으로 설정된 휴일의 기준을 변경하는 함수
        
#         Args:
#             timestep_criteria (dictionary)): 
#                 - "step", "label" 두 개의 key를 갖고 있다.
#                 - EX) {"step":["Sat", "Sun"], "label":["Holiday"]}은 "Sat", "Sun"이 "Holiday"라는 의미이다.
#         """
#         self.holiday_criteria = holiday_criteria

#     def make_day_column(self): # make_day_info
#         """
#         데이터의 시간에 따라 "Day" column에 요일 정보를 추가하는 함수

#         Returns:
#             요일 정보를 포함한 데이터
#         """
#         days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
#         day_list = [days[x] for x in self.data.index.weekday]
#         self.data["Day"] = day_list
#         return self.data
    
#     # Data Public Holiday Create
#     def make_holiday_column(self): # make_holiday_info
#         """
#         데이터의 시간에 따라 "HoliDay" column에 휴일 정보를 추가하는 함수
#         요일 정보를 생성해주는 함수를 활용해서 주말 정보를 휴일로 구분
#         휴일은 설정된 휴일 기준으로 한다.

#         Returns: 
#             요일, 휴일 정보를 포함한 데이터
#         """
#         self.data = self.make_day_column()
        
#         ## public_holiday 생성
#         years = range(self.data.index.min().year, self.data.index.max().year+1)
#         public_holiday_list = []
#         for year in years:
#             public_holiday_list += [x.strftime("%Y-%m-%d") for x in pytimekr.holidays(year)]
        
#         ## 주말 list 생성
#         input_holiday_list = []
#         for input_holiday in self.holiday_criteria["step"]:
#            input_holiday_list += [x.strftime("%Y-%m-%d") for x in self.data[self.data.Day == input_holiday].index.tz_convert(None)]
#         input_holiday_set = set(input_holiday_list)
        
#         ## 최종 휴일 Column 생성 (공휴일 + holiday_criteria 기준)
#         final_holiday_list = public_holiday_list + list(input_holiday_set)
#         holidays = ["holiday" if x.strftime("%Y-%m-%d") in final_holiday_list else "notHoliday" for x in self.data.index]
#         self.data["HoliDay"] = holidays

#         return self.data
    
#     def get_result(self):
#         """
#         Holiday &Not Holiday 에 따른 데이터의 평균 값을 Meta 로 생성하는 함수

#         - 휴일과 휴일이 아닌 날에 따른 분석을 위해 make_holiday_column 함수로 추출한 "HoliDay" column을 활용
        
#         Returns:
#             Holiday, Not Holiday 의 평균 값을 포함한 Dictionary Meta
#         """
#         self.data = self.make_holiday_column()
#         print(">>>>> make holiday column success <<<<<")
#         meanbyholiday_result_dict = self.data.groupby("HoliDay").mean().to_dict()
#         print(">>>>> holiday groupby success <<<<<")
#         meanbyholiday_result_dict = BasicTool.data_none_error_solution(self.labels, meanbyholiday_result_dict)
        
#         return meanbyholiday_result_dict