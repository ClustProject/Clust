from pytimekr import pytimekr
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
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

def get_holidayCycleSet_from_dataframe(data):
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
        if "holiday" in cycle_data["Holiday"][0]:
            cycle_data = cycle_data.drop(["Day", "Holiday"], axis=1)
            holiday_data_list.append(cycle_data)
        else:
            cycle_data = cycle_data.drop(["Day", "Holiday"], axis=1)
            notholiday_data_list.append(cycle_data)
    
    holiday_cycle_set = {"holiday":holiday_data_list, "notholiday":notholiday_data_list}
    return holiday_cycle_set

def get_holidayCycleSet_from_dataset(dataset, feature_list):
    holiday_cycle_set_from_dataset = {}
    for feature in feature_list:
        feature_dataset = dataset[feature]
        holiday_cycle_set_from_dataset[feature] = list(map(get_holidayCycleSet_from_dataframe, feature_dataset))
    
    return holiday_cycle_set_from_dataset
