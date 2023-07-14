import sys
sys.path.append("../..")
sys.path.append("../../../")
import pandas as pd
import datetime, math

from Clust.clust.pipeline import param, data_pipeline

from Clust.clust.ML.clustering.interface import clusteringByMethod
from Clust.clust.tool.plot import plot_interface
from Clust.clust.ML.tool import util


def get_preprocessing_test_pipeline(pipeline_case_param, uncertain_error_to_NaN_flag, param):
    """
    공기질 시나리오의 Processing Data 단계를 pipeline case 별 진행하는 interface

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
    공기질 시나리오의 Processing 단계에서 쓰이는 Clustering으로 기존 Som Cluster에 imputation&smoothing preprocessing 처리를 더한 모듈
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
    result_df = pd.DataFrame()
    for clust_class in clust_class_list:
        for ms_name, class_value in clust_result.items():
            if class_value == str(clust_class):
                result_df = pd.concat([result_df, data[ms_name]])
    return result_df

def select_preprocessed_data_result(data):
    result_df = pd.DataFrame()
    for name in processing_data:
        result_df = pd.concat([result_df, data[name]])
    return result_df
    

def get_univariate_df_by_integration(result_df, start_time, feature, frequency):
    result_df.columns = [feature]
    time_index = pd.date_range(start=start_time, freq = str(frequency)+"T", periods=len(result_df))
    result_df.set_index(time_index, inplace = True)
    
    return result_df

