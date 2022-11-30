import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from Clust.clust.meta.analysisMeta.basicTool import BasicTool

class Statistics():
    def __init__(self, data):
        self.data = data
        self.labels = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

    def get_result(self):
        """
        데이터의 통계적 분포 정보를 Dictionary로 생성하는 함수

        Returns:
            데이터의 통계적 분포 정보를 담고 있는 Dictionary
        """
        statistics_result_dict = self.data.describe().to_dict()
        statistics_result_dict = BasicTool.data_none_error_solution(self.labels, statistics_result_dict)
        return statistics_result_dict