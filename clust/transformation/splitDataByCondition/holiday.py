from pytimekr import pytimekr 
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

def add_holiday_feature(data):
    """    
    A function that adds a new holiday column by designating holidays and weekends as holiday days during the data period.

    Args:
        data (_pd.dataframe_) : Time series data

    Returns:
        _pd.dataframe_ : data(Time sereis data with "Holiday" column)

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

def split_data_by_holiday(data):
    """
    Split the data by holiday/non-holiday.

    Args:
        data (_pd.dataframe_) : Time series data

    Returns:
        _Dictionary_ : split_data_by_holiday(Return value composed of dataframes divided according to each label of holiday and notholiday)

    """
    # Get data with holiday feature
    data = add_holiday_feature(data)
    
    # Split Data by Holiday
    split_data_by_holiday = {}
    split_data_by_holiday["holiday"] = data["holiday" == data["Holiday"]].drop(["Day", "Holiday"], axis=1)
    split_data_by_holiday["notholiday"] = data["notholiday" == data["Holiday"]].drop(["Day", "Holiday"], axis=1)
    
    return split_data_by_holiday

