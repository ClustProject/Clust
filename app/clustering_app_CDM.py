from Clust.clust.pipeline import data_pipeline
from Clust.clust.ML.clustering.interface import clusteringByMethod
import math
import pandas as pd
from Clust.clust.tool.plot import plot_interface
from Clust.clust.ML.tool import util


def get_summarized_clustering_reulst(cluster_num, processing_data, result_y):
    from Clust.clust.ML.tool import util
    data_name = list(processing_data.columns)
    result_dic = util.get_dict_from_two_array(data_name, result_y)

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

def get_clustering_result(processing_data, cluster_num):
    """
    Processing 단계에서 쓰이는 Clustering으로 기존 Som Cluster에 scaling 전처리를 더한 모듈
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
    plt1.show()
    
    y_df = pd.DataFrame(result)
    plt2 = plot_interface.get_graph_result('plt', 'histogram', y_df)
    plt2.show()
    result_dic, clustering_result_by_class = get_summarized_clustering_reulst(cluster_num, processing_data, result)

    return result_dic, clustering_result_by_class


def select_clustering_data_result(data, clust_class_list, clust_result, scaling = False):
    """clustering result에서 선택된클래스 정보만 전달한다.

    Args:
        data (pd.DataFrame): Original Data
        clust_class_list (list): selected class
        clust_result (list): clustering result
        scaling (bool, optional): scaling option,  Defaults to False.

    Returns:
        result_df (pd.DataFrame): dataframe result
    """
    
    result_df = pd.DataFrame()
    # make total scaler
    if scaling == True:
        from sklearn.preprocessing import MinMaxScaler
        result_df_new = pd.DataFrame(pd.Series(data.values.ravel('F')))
        scaler_total = MinMaxScaler()
        scaler_total.fit_transform(result_df_new)
        
        for clust_class in clust_class_list:
            for ms_name, class_value in clust_result.items():
                if class_value == str(clust_class):
                    scaler_indi = MinMaxScaler()
                    scaled_data_indi = scaler_indi.fit_transform(data[[ms_name]])
                    inverse_scaled_data_indi_temp = scaler_total.inverse_transform(scaled_data_indi)
                    result_df[ms_name] =  inverse_scaled_data_indi_temp.reshape(-1)
         
    else:
        for clust_class in clust_class_list:
            for ms_name, class_value in clust_result.items():
                if class_value == str(clust_class):
                     result_df[ms_name] =   data[ms_name]

                    
    return result_df