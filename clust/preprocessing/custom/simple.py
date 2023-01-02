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
    

def preprocessing_basic_for_clust_multiDataSet(dataSet, min_max_from_db, timedelta_frequency_sec):
    """
        simple preprocessing with multiple dataset
    Args:
        dataSet (dictionary consisting of multiple dataFrame): original data
        min_max_from_db (mongo instance): min max information of data
        db_name (string): db_name of data
        timedelta_frequency_sec (timedelta): frequency information
        dataSet (dictionary consisting of multiple dataFrame): original

    Returns:
        dataSet_pre (dictionary consisting of multiple dataFrame): preprocessed dataSet, each dataframe has same length, same frequency without certain error

    """

    # dataSet 형태기 때문에 dataSet형태의 전처리가 필요함
    #필수적인 오류 데이터에 대해서 NaN 처리함
    from Clust.clust.preprocessing.dataPreprocessing import DataProcessing
    CertainParam= {'flag': True, 'data_min_max_limit':min_max_from_db}
    refine_param = {'removeDuplication': {'flag': True}, 'staticFrequency': {'flag': True, 'frequency': timedelta_frequency_sec}}
    outlier_param ={
        "certainErrorToNaN":CertainParam, 
        "unCertainErrorToNaN":{'flag': False}
    }
    imputation_param = {"flag":False}
    process_param = {'refine_param':refine_param, 'outlier_param':outlier_param, 'imputation_param':imputation_param}

    partialP = DataProcessing(process_param)
    dataSet_pre = partialP.multiDataset_all_preprocessing(dataSet)

    return dataSet_pre

def preprocessing_basic_for_clust_oneData(data, mongo_client, db_name, timedelta_frequency_sec):
    """
        simple preprocessing with multiple dataset
    Args:
        data (DataFrame) : original
        mongo_client (mongo instance): instance of meta
        db_name (string): db_name of data
        timedelta_frequency_sec (timedelta): frequency information
        dataSet (dictionary consisting of multiple dataFrame): original

    Returns:
        dataSet_pre (dictionary consisting of multiple dataFrame): preprocessed dataSet

    """
    # dataSet 형태기 때문에 dataSet형태의 전처리가 필요함

    from Clust.clust.ingestion.mongo import customModules
    
    #db에서 가져온 데이터로 만든 민맥스
    
    min_max_from_db = customModules.get_min_max_info_from_bucketMeta(mongo_client, db_name)

    #필수적인 오류 데이터에 대해서 NaN 처리함
    from Clust.clust.preprocessing.dataPreprocessing import DataProcessing
    CertainParam= {'flag': True, 'data_min_max_limit':min_max_from_db}
    refine_param = {'removeDuplication': {'flag': True}, 'staticFrequency': {'flag': True, 'frequency': timedelta_frequency_sec}}
    outlier_param ={
        "certainErrorToNaN":CertainParam, 
        "unCertainErrorToNaN":{'flag': False}
    }
    imputation_param = {"flag":False}
    process_param = {'refine_param':refine_param, 'outlier_param':outlier_param, 'imputation_param':imputation_param}

    partialP = DataProcessing(process_param)
    result = partialP.all_preprocessing(data)

    return result