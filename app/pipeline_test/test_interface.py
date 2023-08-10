import sys
sys.path.append("../..")
sys.path.append("../../../")
import pandas as pd
import datetime, math

from Clust.clust.pipeline import data_pipeline

from Clust.clust.ML.clustering.interface import clusteringByMethod
from Clust.clust.tool.plot import plot_interface
from Clust.clust.ML.tool import util


def get_DFSet_preprocessing_test_pipeline(pipeline_case_param, uncertain_error_to_NaN_flag, param):
    """
    Data Set input과 파라미터 값을 의거하여 유효한 테스트 파이프라인 파라미터를 생성함
    Args:
        preprocessing_case_param(dict)
        uncertain_error_to_NaN_flag (str)
        param (str)test_pipe_param

        
    Example:
        >>> processing_case_param = {
        ... "processing_task_list" : processing_task_list, 
        ... "data_min_max" : data_min_max,
        ... "processing_freq" : processing_freq,
        ... "feature_name" : feature_name,
        ... }
    """
    data_min_max = pipeline_case_param["data_min_max"]
    processing_freq = pipeline_case_param["processing_freq"]
    feature_name = pipeline_case_param["feature_name"]
    timedelta_frequency_min = datetime.timedelta(minutes= processing_freq)
    
    
    processing_task_list = pipeline_case_param["processing_task_list"]

    print("featureName", feature_name)
    param['data_refinement']['static_frequency']['frequency'] = timedelta_frequency_min
    param['data_integration']['integration_param']['integration_frequency'] = timedelta_frequency_min
    param['data_integration']['integration_param']['feature_name'] = feature_name
    param['data_outlier']['certain_error_to_NaN']['data_min_max_limit'] = data_min_max
    param['data_outlier']['uncertain_error_to_NaN']['flag'] = uncertain_error_to_NaN_flag
        
    pipeline = []
    for procssing_task in processing_task_list:
        pipeline.append([procssing_task, param[procssing_task]])
    valid_flag = data_pipeline.pipeline_connection_check(pipeline, input_type = 'DFSet')
    
    if valid_flag:
        return pipeline
    else:
        print("It's not working")
        return None

############# Clustering
def get_clustering_test_result(processing_data, cluster_num):
    """
Processing 단계에서 쓰이는 Clustering으로 기존 Som Cluster에 imputation&smoothing preprocessing 처리를 더한 모듈
    """
    data_scaling_param = {'flag': True, 'method':'minmax'} 
    clustering_clean_pipeline = [['data_scaling', data_scaling_param]]

    ## 1.2. Get clustering input data by cleaning
    clustering_input_data = data_pipeline.pipeline(processing_data, clustering_clean_pipeline)

    # 2. Clustering
    ## 2.1. Set clustering param
    parameter = {
        "method": "som",
        "param": {
            "epochs":5000,
            "som_x":int(math.sqrt(cluster_num)),
            "som_y":int(cluster_num / int(math.sqrt(cluster_num))),
            "neighborhood_function":"gaussian",
            "activation_distance":"euclidean"
        }
    }

    ## 2.2. Start Clustering
    model_path = "model.pkl"
    x_data_series, result, plt1= clusteringByMethod(clustering_input_data, parameter, model_path)

    y_df = pd.DataFrame(result)
    plt2 = plot_interface.get_graph_result('plt', 'histogram', y_df)

    data_name = list(processing_data.columns)
    result_dic = util.get_dict_from_two_array(data_name, result)

    plt1.show()
    plt2.show()

    ## clust class 별 존재하는 데이터 이름과 개수를 출력
    clustering_result_by_class = {}
    for num in range(cluster_num):
        name_list = []
        for name, c_value in result_dic.items():
            if str(num) == c_value:
                name_list.append(name)
        class_num = len(name_list)
        unique_name_list= set([n.split("/")[0] for n in name_list])
        clustering_result_by_class[num] = [unique_name_list, class_num]

    return result_dic, clustering_result_by_class

def select_clustering_data_result(data, clust_class_list, clust_result):
    import pandas as pd 
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    # make total scaler
    result_df = pd.DataFrame(pd.Series(data.values.ravel('F')))
    scaler_total = MinMaxScaler()
    scaler_total.fit_transform(result_df)
    
    inverse_scaled_data_indi= np.array([])
    for clust_class in clust_class_list:
        for ms_name, class_value in clust_result.items():
            if class_value == str(clust_class):
                scaler_indi = MinMaxScaler()
                scaled_data_indi = scaler_indi.fit_transform(data[[ms_name]])
                import matplotlib.pyplot as plt
                inverse_scaled_data_indi_temp = scaler_total.inverse_transform(scaled_data_indi)
                
                inverse_scaled_data_indi_temp = inverse_scaled_data_indi_temp.reshape(-1)
                inverse_scaled_data_indi = np.concatenate((inverse_scaled_data_indi,inverse_scaled_data_indi_temp),axis=0)
    result_df = pd.DataFrame(inverse_scaled_data_indi)   
    return result_df



