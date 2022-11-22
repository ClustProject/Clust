import numpy as np

class BasicTool():
    def data_none_error_solution(labels, result_dict): # 이름 변경 요망
        for label in labels:
            for key in result_dict.keys():
                values = result_dict[key]
                if label not in values.keys(): # 없는 label 값을 None 채우기
                    values[label] = "None"
                if values[label] != "None":
                    if np.isnan(values[label]): # nan -> None
                        values[label] = "None"
                    
                result_dict[key] = values
        return result_dict