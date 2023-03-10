from pytimekr import pytimekr
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import pandas as pd
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

def split_data_by_holiday(data_input):
    """
    Split the data by holiday/non-holiday.

    Args:
        input (dataframe, dictionary): input data

    Returns:
        dictionary: Return value composed of dataframes divided according to each label of holiday and notholiday.
    """
       
    def _split_data_by_holiday_from_dataframe(data):
        """
        Split the data by holiday/non-holiday.

        Args:
            data (dataframe): Time series data

        Returns:
            dictionary: Return value composed of dataframes divided according to each label of holiday and notholiday.
        """
        # Get data with holiday feature
        data = get_holiday_feature(data)
        
        # Split Data by Holiday
        split_data_by_holiday = {}
        split_data_by_holiday["holiday"] = data["holiday" == data["Holiday"]].drop(["Day", "Holiday"], axis=1)
        split_data_by_holiday["notholiday"] = data["notholiday" == data["Holiday"]].drop(["Day", "Holiday"], axis=1)
        
        return split_data_by_holiday

    def _split_data_by_holiday_from_dataset(dataset):
        """
        Split Data Set by holiday/non-holiday.

        Args:
            dataset (Dictionary): dataSet, dictionary of dataframe (ms data). A dataset has measurements as keys and dataframes(Timeseries data) as values.
        
        Returns:
            dictionary: Return value has measurements as keys and split result as values. 
                        split result composed of dataframes divided according to each label of holiday and notholiday.
        """
        split_dataset_by_holiday = {}
        for ms_name in dataset:
            data = dataset[ms_name]
            if not(data.empty):
                split_data_by_holiday_dict = _split_data_by_holiday_from_dataframe(data)
                split_dataset_by_holiday[ms_name] = split_data_by_holiday_dict

        return split_dataset_by_holiday

    
    if isinstance(data_input, dict):
        result = _split_data_by_holiday_from_dataset(data_input)
    elif isinstance(data_input, pd.DataFrame):
        result = _split_data_by_holiday_from_dataframe(data_input)
    
    return result
        
 