import pandas as pd
import numpy as np

#### Duration 
def getRobustScaledDF(DF):
    from sklearn.preprocessing import RobustScaler
    transformer = RobustScaler()
    scaledDF = pd.DataFrame(transformer.fit_transform(DF), index = DF.index, columns = DF.columns )
    return scaledDF

def get_corrMatrix(data):
    corr_matrix = data.corr(method='pearson').values.tolist()
    return corr_matrix


def checkNumericColumns(data, checkColumnList=None):
    """
    This function returns data by trnsforming the Numeric type colums specified in "checkColumnList". 
    If checkColumnList is None, all columns are converted to Numeric type.

    Args:
        data (dataFrame): input Data
        checkColumnList (string array or None): db_name

    Returns:
        dataSet: dataSet

    1. CheckColumnList==None : change all columns to numeric type
    2. CheckColumnList has values: change only CheckColumnList to numeric type
    """
    if checkColumnList:
        pass
    
    else:
        checkColumnList = list(data.select_dtypes(include=['object']).columns)

    data[checkColumnList] = data[checkColumnList].apply(pd.to_numeric, errors='coerce')
    
    return data

def nan_to_none_in_dict(labels, analysis_result_dict): # 이름 변경 요망 
    """
    This function is necessary to create a meta stored in MongoDB.
    This function changes nan of all values to "none" and fills the Label value by saving "None" to a Label that does not exist because there is no information.
    
    The analysis_result_dict stored in MongoDB must be encoded as Json. However, since Nan is not a valid Json symbol, it must be changed to "None" when Nan exists.
    And if the label does not exist, it is difficult to utilize the analysis result.
    Label means the key of each corresponding analysis result.

    Args:
        labels (list): Label means the key of each corresponding analysis result.
        analysis_result_dict (dictionary): analysis result
        
    Example:
    
    >>> labels = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    
    >>> analysis_result_dict = {
                            'in_co2': {
                                    'count': 329669.0,
                                    'mean': 500.6160906848991,
                                    'std': 132.47911810569934,
                                    'min': 243.0,
                                    '25%': 419.0,
                                    '50%': 465.0,
                                    '75%': 550.0,
                                    'max': 1707.0 } }

    Returns:
        dictionary: analysis result
        
    """
    for label in labels:
        for key in analysis_result_dict.keys():
            values = analysis_result_dict[key]
            if label not in values.keys(): # 없는 label 값을 None 채우기
                values[label] = "None"
            if values[label] != "None":
                if np.isnan(values[label]): # nan -> None
                    values[label] = "None"
                
            analysis_result_dict[key] = values
    return analysis_result_dict