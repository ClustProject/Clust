import sys
import pandas as pd
import datetime
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from Clust.clust.transformation.splitDataByCycle import dataByCycle
from Clust.clust.quality.NaN import clean_feature_data



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