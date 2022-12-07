import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import pandas as pd
from Clust.clust.meta.analysisMeta.basicTool import BasicTool

class StatisticsAnalysis():
    """
    Analyzer
    데이터를 통계적 관점에서 분석한 결과 생성
    데이터를 통계적 관점에서 분석하는 분석기로 2가지 방식이 있음
        1. Pandas에서 제공하는 기본적인 통계 분석
        2. 데이터 Feature별 기준 범주 및 라벨에 따른 분포를 분석
    """
    def __init__(self, data):
        self.data = data

    def get_basic_analysis_result(self):
        """
        Analyze statistical distribution information of data

        Returns:
            Dictionary : Analysis Result 
        
        
        
        데이터의 통계적 분포 정보를 Dictionary로 생성하는 함수
        
        데이터의 통계적 분포 정보를 분석

        Returns:
            데이터의 통계적 분포 정보를 담고 있는 Dictionary
        """
        labels = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        statistics_result_dict = self.data.describe().to_dict()
        statistics_result_dict = BasicTool.data_none_error_solution(labels, statistics_result_dict)
        return statistics_result_dict
    
    def get_count_by_label_analysis_result(self, base_meta):
        """
        Analyze statistical distribution information of data

        Args:
            bk_name (string): bucket(databse)

        Returns:
            List: measurement list
        
        데이터의 Label Information Meta 를 생성하는 함수

        - 데이터베이스에 Label Information 정보가 있어야 함

        """
        data_cut = pd.DataFrame()
        countbyfeaturelabel_result_dict = {}
        for column_info in base_meta["columnInformation"]:
            column = column_info["columnName"]
            if "columnLevelCriteria" not in column_info.keys():
                countbyfeaturelabel_result_dict[column] = ["None"]
            else:
                if column in self.data.columns: 
                    data_cut[column] = pd.cut(x=self.data[column], 
                                        bins=column_info["columnLevelCriteria"]["step"],
                                        labels=column_info["columnLevelCriteria"]["label"])
                    labelcount = dict(data_cut[column].value_counts())
                    label_dict = {}
                    label_ls = []
                    
                    for n in range(len(labelcount)):
                        label_dict["value"] = int(labelcount[list(labelcount.keys())[n]])
                        label_dict["name"] = list(labelcount.keys())[n]
                        label_ls.append(label_dict.copy())

                    countbyfeaturelabel_result_dict[column] = label_ls
                
        return countbyfeaturelabel_result_dict