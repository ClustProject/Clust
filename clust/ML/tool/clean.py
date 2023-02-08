import sys
import pandas as pd
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from Clust.clust.transformation.purpose import machineLearning as ML
from Clust.clust.transformation.general.dataScaler import DataScaler
from Clust.clust.transformation.splitDataByCycle import dataByCycle
from Clust.clust.quality.NaN import clean_feature_data
from Clust.clust.quality.NaN import cleanData


def delete_low_quality_train_val_data(train, val, clean_train_data_param, integration_freq_sec, nan_processing_param):
    print("------", nan_processing_param)
    if clean_train_data_param =='Clean':
        import datetime
        # TODO integration_freq sec  사용을 안하는데 추후 문제될 수 있으니 확인해봐야 함
        #timedelta_frequency_sec = datetime.timedelta(seconds= integration_freq_sec)
        # 3. quality check
        nan_infor_clean_data = nan_processing_param['NanInfoForCleanData']
        print("------", nan_infor_clean_data)
        CMS = cleanData.CleanData()
        train = CMS.get_cleanData_by_removing_column(train, nan_infor_clean_data) 
        val = CMS.get_cleanData_by_removing_column(val, nan_infor_clean_data) 

    else:
        pass
    return train, val


def clean_nan_df(data_set, nan_processing_param, timedelta_frequency_sec):

    feature_cycle=nan_processing_param['feature_cycle']
    feature_cycle_times=nan_processing_param['feature_cycle_times']
    nan_infor_clean_data=nan_processing_param['NanInfoForCleanData']

    feature_list = data_set.columns
    day_cycle = dataByCycle.getCycleSelectDataSet(data_set, feature_cycle, feature_cycle_times, timedelta_frequency_sec)

    CMS = clean_feature_data.CleanFeatureData(timedelta_frequency_sec)
    filter_imputed_data = CMS.getMultipleCleanDataSetsByFeature(day_cycle, nan_infor_clean_data, None) 
    clean_data = pd.concat(filter_imputed_data.values())

    return clean_data


def get_cleand_data(data, clean_train_data_param, integration_freq_sec, nan_processing_param):
    if clean_train_data_param =='Clean':
        import datetime
        timedelta_frequency_sec = datetime.timedelta(seconds= integration_freq_sec)
        result = p3.clean_nan_df(data, nan_processing_param,  timedelta_frequency_sec)

    else:
        result = data.copy()
        pass
    
    return result