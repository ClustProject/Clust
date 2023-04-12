import pandas as pd
import sys
sys.path.append("../")
import math
from Clust.clust.tool.stats_table import metrics
from Clust.clust.ML.tool import model as model_manager


# KETIAppdataServer/dataDomainExploration
# KETIAppTestCode/Domain, Cycle Data
# 기타 EDA에서 활용되고 있음
def get_somClustering_result_from_dataSet(data_set, feature_name, min_max, timedelta_frequency_min, duration, NaNProcessingParam, model_type, cluster_num):

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
    process_param = processing_interface.get_default_processing_param(min_max, timedelta_frequency_min)

    # 2. one DF preparation
    from Clust.clust.transformation.general import select_interface
    select_param ={"feature_name":feature_name, "duration":duration, "frequency":timedelta_frequency_min }
    data_DF = select_interface.get_data_result("oneDF_with_oneFeature_from_multipleDF", data_set, select_param)

    # 3. quality check
    from Clust.clust.quality.NaN import cleanData
    CMS = cleanData.CleanData()
    data = CMS.get_cleanData_by_removing_column(data_DF, NaNProcessingParam) 

    figdata=None
    result_dic={}
    if len(data.columns) > 1:
        # 4. preprocessing for clustering
        from Clust.clust.preprocessing import processing_interface
        imputation_param = {
            "flag":True,
            "imputation_method":[{"min":0,"max":300,"method":"linear", "parameter":{}}, 
                                {"min":0,"max":10000,"method":"mean", "parameter":{}}],
            "totalNonNanRatio":1 }
        data = processing_interface.get_data_result('imputation', data, imputation_param)
        process_param={'flag': True, 'emw_param':0.3}
        data = processing_interface.get_data_result('smoothing', data, process_param)
        process_param={'flag': True, 'method':'minmax'} 
        data = processing_interface.get_data_result('scaling', data, process_param)
        
        """
        # SOM
        """

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
        result, figdata= clusteringByMethod(data, parameter, model_path)
        
        # histogram by label
        from Clust.clust.tool.plot import plot_interface
        y_df = pd.DataFrame(result)
        plt2 = plot_interface.get_graph_result('plt', 'histogram', y_df)
        #plt2.show()
        
        from Clust.clust.ML.tool import util
        data_name = list(data.columns)
        result_dic = util.get_dict_from_two_array(data_name, result)

    return result_dic, figdata




