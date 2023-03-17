import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocessing_smoothing_scaling(data, ewm_parameter=0.3):
    """

    Preprocessing (Smoothing and Scaling)

    Args:
        data (dataFrame): input data
        ewm_parameter (float, optional): emw parameter. Defaults to 0.3.


    Returns:
        data (dataFrame): preprocessed result
    """
    data = data.ewm(com=ewm_parameter).mean()
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=list(data.columns), index = data.index)      
    return data
    

def preprocessing_basic_for_clust(dataSet, min_max, timedelta_frequency_sec):
    """
        simple preprocessing with multiple dataset
        Output multiple data (value of dataSet) -> same length, same description frequency, no certain error
    Args:
        dataSet (dictionary consisting of multiple dataFrame): original data
        min_max (dict): min max information of data
        timedelta_frequency_sec (timedelta): frequency information

    Returns:
        dataSet_pre (dictionary consisting of multiple dataFrame): preprocessed dataSet, each dataframe has same length, same frequency without certain error

    """


    #필수적인 오류 데이터에 대해서 NaN 처리함
    CertainParam= {'flag': True, 'data_min_max_limit':min_max}
    refine_param = {'removeDuplication': {'flag': True}, 'staticFrequency': {'flag': True, 'frequency': timedelta_frequency_sec}}
    outlier_param ={
        "certainErrorToNaN":CertainParam, 
        "unCertainErrorToNaN":{'flag': False}
    }
    imputation_param = {"flag":False}
    process_param = {'refine_param':refine_param, 'outlier_param':outlier_param, 'imputation_param':imputation_param}

    from Clust.clust.preprocessing import processing_interface
    dataSet_pre = processing_interface.get_data_result('all', dataSet , process_param)
        
    return dataSet_pre
