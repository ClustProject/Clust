import pandas as pd
import sys
sys.path.append("../")

from Clust.clust.tool.stats_table import metrics
from Clust.clust.ML.tool import model as model_manager
from Clust.clust.ML.common.common import p2_dataSelection as p2
from Clust.clust.ML.common.common import p4_testing as p4
from Clust.clust.ML.forecasting.test import ForecasatingTest as FT
from Clust.clust.ML.forecasting.inference import ForecastingInfernce as FI


# Clust/KETIAppTestCode/KWeather2nd/20-01, 20-02, 20-03 Clust/clust/example/4-1, 4-2 테스트 코드와 관련 있음
# 서버의 dataDomainExploration.py도 연결
def get_somClustering_result_from_dataSet(data_set, feature_name, min_max, timedelta_frequency_min, duration, NaNProcessingParam):

    """_customized clustering function_ 
        1) preprocessing for making one DF
        2) one DF preparation
        3) quality check to remove bad quality data
        4) preprocessing for clustering
        5) clustering

    Args:
        data_set (dict): input dataset (key: dataName, value:data(dataFrame)) 
        feature_name (str): 추출한 feature 이름
        min_max (dict):전처리 중 outlier 제거를 위해 활용할 min, max 값
        timedelta_frequency_min (timedelta): 데이터의 기술 주기
        duration (dict): 데이터의 유효한 시간 구간
        NaNProcessingParam (dict) : 데이터 퀄러티 체크를 위한 정보
        fig_width (int): 최종 결과 그림을 위한 크기 정보 
        fig_height (int): 최종 결과 그림을 위한 크기 정보

    Returns:
        _type_: _description_
    """
    # 1. preprocessing for oneDF
    from Clust.clust.preprocessing.custom import simple
    dataSet_pre = simple.preprocessing_basic_for_clust_multiDataSet(data_set, min_max, timedelta_frequency_min)

    # 2. one DF preparation
    from Clust.clust.transformation.general import dataframe
    data_DF = dataframe.get_oneDF_with_oneFeature_from_multipleDF(dataSet_pre, feature_name, duration, timedelta_frequency_min)

    # 3. quality check
    from Clust.clust.quality.NaN import cleanData
    CMS = cleanData.CleanData()
    data = CMS.get_cleanData_by_removing_column(data_DF, NaNProcessingParam) 

    # 4. preprocessing for clustering
    from Clust.clust.preprocessing.custom.simple import preprocessing_smoothing_scaling
    data = preprocessing_smoothing_scaling(data, ewm_parameter=0.3)

    # 5. clustering
    from Clust.clust.tool.plot import plot_features
    # plot_features.plot_all_column_data_in_sub_plot(data, fig_width, fig_height, fig_width_num = 4)
    

    
    """
    # SOM
    """
    parameter = {
        "method": "som",
        "param": {
            "epochs":5000,
            "som_x":2,
            "som_y":2,
            "neighborhood_function":"gaussian",
            "activation_distance":"euclidean"
        }
    }
    
    """
    # K-Means
    parameter = {
         "method": "kmeans",
         "param": {
             "n_clusters": 3,
             "metric": "euclidean"
         }
    }
    """


    from Clust.clust.ML.clustering.interface import clusteringByMethod
    model_path = "model.pkl"
    result, figdata= clusteringByMethod(data, parameter, model_path)

    return result, figdata








# --------------------------------------- forecasting / test & inference ---------------------------------------
def get_forecasting_test(data_name, data_meta, model_meta, data_folder_name, db_client=None):

    data_save_mode = data_meta[data_name]["integrationInfo"]["DataSaveMode"]
    data = p2.get_saved_integrated_data(data_save_mode, data_name, data_folder_name)
    
    scaler_file_path = model_meta['files']['scalerFile']["filePath"]
    model_file_path = model_meta['files']['modelFile']["filePath"]

    feature_list = model_meta["featureList"]
    target_col = model_meta['transformParameter']["target_col"]
    scaler_param = model_meta["scalerParam"]
    model_method = model_meta["model_method"]
    train_parameter = model_meta["trainParameter"]
    transform_parameter = model_meta["transformParameter"]
    integration_freq_sec = model_meta['trainDataInfo']["integration_freq_sec"]
    clean_train_data_param = model_meta["cleanTrainDataParam"] 
    nan_processing_param = model_meta['NaNProcessingParam']

    test_data, scaler = p4.get_scaled_test_data(data[feature_list], scaler_file_path, scaler_param)
    clean_test_data = p4.get_cleand_data(test_data, clean_train_data_param, integration_freq_sec, nan_processing_param)

    ft = FT()
    ft.set_param(model_meta)
    ft.set_data(clean_test_data)
    model = model_manager.load_pickle_model(model_file_path)
    preds, trues = ft.get_result(model)

    df_result = p4.get_prediction_df_result(preds, trues, scaler_param, scaler, feature_list, target_col)
    df_result.index = test_data[(transform_parameter['future_step']+transform_parameter['past_step']-1):].index
    result_metrics =  metrics.calculate_metrics_df(df_result)

    return df_result, result_metrics


def get_forecasting_inference(data, model_meta):

    scaler_file_path = model_meta['files']['scalerFile']["filePath"]
    model_file_path = model_meta['files']['modelFile']["filePath"]

    feature_list = model_meta["featureList"]
    target_col = model_meta['transformParameter']["target_col"]
    scaler_param = model_meta["scalerParam"]
    past_step = model_meta['transformParameter']['past_step']

    feature_data = data[feature_list]
    step_data = feature_data[-past_step:][feature_list].values
    df_data = pd.DataFrame(step_data, columns = feature_list)

    input_data, scaler = p4.get_scaled_test_data(df_data[feature_list], scaler_file_path, scaler_param)

    fi = FI()
    fi.set_param(model_meta)
    fi.set_data(input_data)
    model = model_manager.load_pickle_model(model_file_path)
    preds = fi.get_result(model)


    if scaler_param =='scale':
        base_df_for_inverse= pd.DataFrame(columns=feature_list, index=range(len(preds)))
        base_df_for_inverse[target_col] = preds
        inverse_result = pd.DataFrame(scaler.inverse_transform(base_df_for_inverse), columns=feature_list, index=base_df_for_inverse.index)
        target_data = inverse_result[target_col]
        prediction_result = pd.DataFrame(data={target_col: target_data}, index=range(len(preds)))
        
    else:
        prediction_result = pd.DataFrame(data={target_col: preds}, index=range(len(preds)))


    return prediction_result





# --------------------------------------- regression / test & inference ---------------------------------------






# --------------------------------------- classification / test & inference ---------------------------------------





