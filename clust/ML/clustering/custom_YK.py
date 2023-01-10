def get_somClustering_result_from_dataSet(data_set, feature_name, min_max, timedelta_frequency_min, duration, NanInfoForCleanData, fig_width, fig_height):

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
        NanInfoForCleanData (dict) : 데이터 퀄러티 체크를 위한 정보
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
    data = CMS.get_cleanData_by_removing_column(data_DF, NanInfoForCleanData) 

    # 4. preprocessing for clustering
    from Clust.clust.preprocessing.custom.simple import preprocessing_smoothing_scaling
    data = preprocessing_smoothing_scaling(data, ewm_parameter=0.3)

    # 5. clustering
    from Clust.clust.tool.plot import plot_features
    # plot_features.plot_all_column_data_in_sub_plot(data, fig_width, fig_height, fig_width_num = 4)
    

    # SOM
    # parameter = {
    #     "method": "som",
    #     "param": {
    #         "epochs":50000,
    #         "som_x":2,
    #         "som_y":2,
    #         "neighborhood_function":"gaussian",
    #         "activation_distance":"euclidean"
    #     }
    # }
    
    # K-Means
    parameter = {
        "method": "kmeans",
        "param": {
            "n_clusters": 3,
            "metric": "euclidean"
        }
    }

    from Clust.clust.ML.clustering.interface_YK import clusteringByMethod
    result, figdata= clusteringByMethod(data, parameter)

    return result, figdata
