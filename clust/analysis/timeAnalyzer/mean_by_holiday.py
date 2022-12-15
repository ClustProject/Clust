from unittest import result
from pytimekr import pytimekr
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from Clust.clust.transformation.general.basicTransform import nan_to_none_in_dict
from Clust.clust.transformation.splitDataByCondition import holiday

def get_mean_analysis_result_by_holiday(data):
    """
    Holiday &Not Holiday 에 따른 데이터의 평균 값을 Meta 로 생성하는 함수

    - 휴일과 휴일이 아닌 날에 따른 분석을 위해 make_holiday_column 함수로 추출한 "HoliDay" column을 활용
    
    Returns:
        Holiday, Not Holiday 의 평균 값을 포함한 Dictionary Meta
    """
    #self.data = self.make_holiday_column()
    labels = ["holiday", "notholiday"]
    data = holiday.get_holiday_feature(data)
    print(">>>>> make holiday column success <<<<<")
    meanbyholiday_result_dict = data.groupby("Holiday").mean().to_dict()
    print(">>>>> holiday groupby success <<<<<")
    meanbyholiday_result_dict = nan_to_none_in_dict(labels, meanbyholiday_result_dict)
    
    return meanbyholiday_result_dict
