def get_clustering_result_from_dataSet(data_set, feature_name, min_max, timedelta_frequency_min, duration, NanInfoForClenData, fig_width, fig_height):

    from Clust.clust.preprocessing.custom import simple
    dataSet_pre = simple.preprocessing_basic_for_clust_multiDataSet(data_set, min_max, timedelta_frequency_min)

    from Clust.clust.transformation.general import dataframe
    dataDF = dataframe.get_oneDF_with_oneFeature_from_multipleDF(dataSet_pre, feature_name, duration, timedelta_frequency_min)

    from Clust.clust.quality.NaN import cleanData
    CMS = cleanData.CleanData()
    data = CMS.get_cleanData_by_removing_column(dataDF, NanInfoForClenData) 

    from Clust.clust.preprocessing.custom.simple import preprocessing_smoothing_scaling
    data = preprocessing_smoothing_scaling(data, ewm_parameter=0.3)

    from Clust.clust.tool.plot_graph import plot_features
    plot_features.plot_all_column_data_inSubPlot(data, fig_width, fig_height, fig_width_num = 4)
    
    from Clust.clust.ML.clustering.interface import clusteringByMethod
    result, figdata, figdata2 = clusteringByMethod(data, 'som',  2, 2)

    return result, figdata, figdata2
