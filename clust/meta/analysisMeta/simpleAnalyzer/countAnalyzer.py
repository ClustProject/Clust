import pandas as pd

class CountByFeatureLabel():
    def __init__(self, data, base_meta):
        self.data = data
        self.base_meta = base_meta
        self.data_cut = pd.DataFrame()
    
     # Data Label Information Meta Create
    def get_result(self):
        """
        데이터의 Label Information Meta 를 생성하는 함수

        - 데이터베이스에 Label Information 정보가 있어야 함

        """
        countbyfeaturelabel_result_dict = {}
        for column_info in self.base_meta["columnInformation"]:
            column = column_info["columnName"]
            if "columnLevelCriteria" not in column_info.keys():
                countbyfeaturelabel_result_dict[column] = ["None"]
            else:
                if column in self.data.columns: 
                    self.data_cut[column] = pd.cut(x=self.data[column], 
                                        bins=column_info["columnLevelCriteria"]["step"],
                                        labels=column_info["columnLevelCriteria"]["label"])
                    labelcount = dict(self.data_cut[column].value_counts())
                    label_dict = {}
                    label_ls = []
                    
                    for n in range(len(labelcount)):
                        label_dict["value"] = int(labelcount[list(labelcount.keys())[n]])
                        label_dict["name"] = list(labelcount.keys())[n]
                        label_ls.append(label_dict.copy())

                    countbyfeaturelabel_result_dict[column] = label_ls
                
        return countbyfeaturelabel_result_dict