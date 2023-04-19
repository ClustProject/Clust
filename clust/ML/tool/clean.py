import sys
import pandas as pd
import datetime
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from Clust.clust.transformation.purpose import machineLearning as ML
from Clust.clust.transformation.general.dataScaler import DataScaler
from Clust.clust.transformation.splitDataByCycle import dataByCycle
from Clust.clust.quality.NaN import clean_feature_data
from Clust.clust.quality.NaN import cleanData

# p3_training
# integration_freq_sec 삭제
def delete_low_quality_train_val_data(train, val, clean_level, nan_processing_param):
    """
    if clean_mode == clean, remove nan data by column

    Args:
        train (dataframe): train data
        val (dataframe): validation data
        clean_mode (string): clean or noclean
        nan_processing_param (dict): feature cycle parameter

    Returns:
        train (dataframe) : train data
        val (dataframe) : validation data
    
    """
    print("------", nan_processing_param)
    if clean_level == 4:
        # TODO integration_freq sec  사용을 안하는데 추후 문제될 수 있으니 확인해봐야 함
        #timedelta_frequency_sec = datetime.timedelta(seconds= integration_freq_sec)
        # 3. quality check
        nan_info_clean_data = nan_processing_param['NanInfoForCleanData']
        print("------", nan_info_clean_data)
        CMS = cleanData.CleanData()
        train = CMS.get_cleanData_by_removing_column(train, nan_info_clean_data) 
        val = CMS.get_cleanData_by_removing_column(val, nan_info_clean_data) 

    else:
        pass
    return train, val

# p3_training
def clean_nan_df(data_set, nan_processing_param, timedelta_frequency_sec):
    """
    

    Args:
        data_set (dataframe): data set
        nan_processing_param (dict): feature cycle parameter
        timedelta_frequency_sec (datetime): time frequency seconds

    Returns:
        clean_data (dataframe) : clean data
    
    """

    feature_cycle=nan_processing_param['feature_cycle']
    feature_cycle_times=nan_processing_param['feature_cycle_times']
    nan_info_clean_data=nan_processing_param['NanInfoForCleanData']

    feature_list = data_set.columns
    day_cycle = dataByCycle.getCycleSelectDataSet(data_set, feature_cycle, feature_cycle_times, timedelta_frequency_sec)

    CMS = clean_feature_data.CleanFeatureData(timedelta_frequency_sec)
    filter_imputed_data = CMS.getMultipleCleanDataSetsByFeature(day_cycle, nan_info_clean_data, None) 
    clean_data = pd.concat(filter_imputed_data.values())

    return clean_data

 # p4_testing
def get_cleand_data(data, clean_level, integration_freq_sec, nan_processing_param):
    """
    if clean_mode == clean, remove nan data by column

    Args:
        train (dataframe): train data
        clean_mode (string): clean or noclean
        integration_freq_sec (int): time frequency seconds
        nan_processing_param (dict): feature cycle parameter

    Returns:
        result (dataframe) : clean data
    
    """
    if clean_level == 4:
        timedelta_frequency_sec = datetime.timedelta(seconds= integration_freq_sec)
        result = clean_nan_df(data, nan_processing_param,  timedelta_frequency_sec)

    else:
        result = data.copy()
        pass
    
    return result