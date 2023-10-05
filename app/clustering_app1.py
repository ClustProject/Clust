import pandas as pd
import sys
sys.path.append("../")
import math
from Clust.clust.pipeline import data_pipeline, param

# KETIAppdataServer/dataDomainExploration
# KETIAppTestCode/Domain, Cycle Data
# 기타 EDA에서 활용되고 있음
def clustering_app_c_1(data_set, feature_name, min_max, timedelta_frequency_min, duration, NaNProcessingParam, model_type, cluster_num):
    # 1-1-1

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
    from Clust.clust.preprocessing import processing_interface
    process_param = processing_interface.clustering_app_t1(min_max, timedelta_frequency_min)

    # 2. preprocessing pipeline
    pipeline = [
        ['data_integration',{'integration_type': 'one_feature_based_integration',
                             'integration_param': {'feature_name': feature_name,
                            'duration': duration,
                            'integration_frequency': timedelta_frequency_min}}],
        ['data_quality_check',{'quality_method': 'data_with_clean_feature','quality_param': {'nan_processing_param': NaNProcessingParam}}],
        ['data_imputation',
            {'flag': True,
            'imputation_method': [{'min': 0,
                'max': 30000,
                'method': 'linear',
                'parameter': {}}],
            'total_non_NaN_ratio': 1}],
        ['data_smoothing', {'flag': True, 'emw_param': 0.3}],
        ['data_scaling', {'flag': True, 'method': 'minmax'}]]
        
    
    data = data_pipeline.pipeline(data_set, pipeline, False)    

    result_dic, plt1 = app_clustering(data, cluster_num )    

    return result_dic, plt1

def app_clustering(data, cluster_num, model_type ='som'):

    """clustering number에 기반하녀 model type을 설정하고 클러스터링을 수행함
    Args:
        data(pd.DataFrame)
        cluster_num(int)
        model_type(str):som, kmeans
        
    Returns:
        result_dic (dict)
        plt1
        plt2
    """
    model_type = 'som'

    if model_type =='som':
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
    elif model_type =='kmeans':  
        # K-Means
        parameter = {
            "method": "kmeans",
            "param": {
                "n_clusters": cluster_num,
                "metric": "euclidean"
            }
        }
        

    from Clust.clust.ML.clustering.interface import clusteringByMethod
    model_path = "model.pkl"
    x_data_series, result, plt1= clusteringByMethod(data, parameter, model_path)    
    
    # histogram by label
    from Clust.clust.tool.plot import plot_interface
    y_df = pd.DataFrame(result)
            
    from Clust.clust.ML.tool import util
    data_name = list(data.columns)

    result_dic = util.get_dict_from_two_array(data_name, result)
    
    return result_dic, plt1
    



