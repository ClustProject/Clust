import sys
sys.path.append("../..")
sys.path.append("../../../")

import math
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from Clust.clust.pipeline import data_pipeline
from Clust.clust.ML.clustering.interface import clusteringByMethod
from Clust.clust.tool.plot import plot_interface
from Clust.clust.ML.tool import util

def pipeline_clustering(processing_data, cluster_num):
    """
    공기질 시나리오의 Processing 단계에서 쓰이는 Clustering으로 기존 Som Cluster에 imputation&smoothing preprocessing 처리를 더한 모듈
    """
    # 1. Imputation & Smoothing
    ## 1.1. Set imputation & smoothing param
    imputation_param = {"flag":True,
                    "imputation_method":[{"min":0,"max":300,"method":"linear", "parameter":{}}, 
                                        {"min":0,"max":10000,"method":"mean", "parameter":{}}],
                    "total_non_NaN_ratio":1 }
    smoothing_param={'flag': True, 'emw_param':0.3}

    clustering_clean_pipeline = [['data_imputation', imputation_param],
                                 ['data_smoothing', smoothing_param]]

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

def get_univariate_df_by_selecting_clustering_data(data, clust_result, start_time, feature, frequency, clust_class_list):
    """
    Clustering 결과에서 선택한 데이터을 기반으로 integration vertical 진행하는 모듈
    """
    result_df = pd.DataFrame()
    for clust_class in clust_class_list:
        for ms_name, class_value in clust_result.items():
            if class_value == str(clust_class):
                result_df = pd.concat([result_df, data[ms_name]])
    result_df.columns = [feature]
    time_index = pd.date_range(start=start_time, freq = str(frequency)+"T", periods=len(result_df))
    result_df.set_index(time_index, inplace = True)
    
    return result_df
