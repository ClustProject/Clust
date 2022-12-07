import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import pandas as pd
from Clust.clust.meta.analysisMeta.basicTool import BasicTool

class StatisticsAnalysis():
    """
    Generate statistical analysis results. There are 2 types.
        1. Basic statistical analysis provided by Pandas
        2. Analyze the distribution according to the criterion category (label) by data column
    
    Args:
        data (dataframe) : Time Series Data
    """
    def __init__(self, data):
        self.data = data

    def get_basic_analysis_result(self):
        """
        Generate the basic statistical analysis results provided by Pandas. 
        The analysis result is information about the statistical distribution of the data.
                
        Returns:
            Dictionary : Analysis Results
            
        Example:
        >>> AnalysisResult = {'column1': {
            'count': 1007583.0,
            'mean': 353.9929951180201,
            'std': 84.57299647078351,
            'min': 177.0,
            '25%': 279.0,
            '50%': 366.0,
            '75%': 413.0,
            'max': 870.0
            }}
        """
        labels = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        statistics_result_dict = self.data.describe().to_dict()
        statistics_result_dict = BasicTool.data_none_error_solution(labels, statistics_result_dict)
        return statistics_result_dict
    
    def get_count_by_label_analysis_result(self, base_meta):
        """
        Generate analysis results for the distribution according to the criterion category (label) for each data column.
        Essentially, Label Information must exist in bucket_meta_info of the measurement.

        Args:
            base_meta (dictionary): bucket_meta_info of the measurement

        Returns:
            Dictionary: Anaysis Results

        Example:
        >>> AnalysisResult = {'column1': [
            {'value': 775053, 'name': '좋음'},
            {'value': 134025, 'name': '보통'},
            {'value': 19865, 'name': '나쁨'},
            {'value': 227, 'name': '매우나쁨'}
            ]}
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