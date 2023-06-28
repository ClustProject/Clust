import sys
sys.path.append("../../../")
sys.path.append("../..")

import math
import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from Clust.clust.pipeline import data_pipeline
from Clust.setting import influx_setting_KETI as ins
from Clust.clust.ingestion.mongo import mongo_client
from Clust.clust.meta.metaDataManager import bucketMeta

from Clust.clust.ML.clustering.interface import clusteringByMethod
from Clust.clust.tool.plot import plot_interface
from Clust.clust.ML.tool import util

def set_case_11_pipeparam(bucket, integration_freq_min, feature_name):
    # 1. refine_param
    mongo_client_ = mongo_client.MongoClient(ins.CLUSTMetaInfo2)
    min_max = bucketMeta.get_min_max_info_from_bucketMeta(mongo_client_, bucket)
    timedelta_frequency_min = datetime.timedelta(minutes= integration_freq_min)

    refine_param = {"remove_duplication": {'flag': True}, 
                    "static_frequency": {'flag': True, 'frequency': timedelta_frequency_min}}

    outlier_param ={
        "certain_error_to_NaN": {'flag': True, 'data_min_max_limit':min_max}, 
        "uncertain_error_to_NaN":{'flag': False}}


    cycle_split_param={
        "split_method":"cycle",
        "split_param":{
            'feature_cycle' : 'Day',
            'feature_cycle_times' : 1}
    }

    integration_param={
        "integration_param":{"feature_name":feature_name, "duration":None, "integration_frequency":timedelta_frequency_min },
        "integration_type":"one_feature_based_integration"
    }
        
    quality_param = {
        "quality_method":"data_with_clean_feature", 
        "quality_param":{"nan_processing_param":{'type':'num', 'ConsecutiveNanLimit':4, 'totalNaNLimit':19}}
    }

    pipeline = [['data_refinement', refine_param],
                ['data_outlier', outlier_param],
                ['data_split', cycle_split_param],
                ['data_integration', integration_param],
                ['data_quality_check', quality_param]]
    
    return pipeline


#def clustering_case_11(processing_data, cluster_num):
def pipeline_clustering(processing_data, cluster_num):
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

    return result_dic

def get_clustering_result_distribution(clust_result, clust_class_list):
    distribution_result = {}
    for clust_class in clust_class_list:
        name_list = []
        for name, c_value in clust_result.items():
            if str(clust_class) == c_value:
                name_list.append(name)
        class_num = len(name_list)
        unique_name_list= set([n.split("/")[0] for n in name_list])
        distribution_result[clust_class] = [unique_name_list, class_num]
    return distribution_result

def get_univariate_df_by_selecting_clustering_data(data, clust_result, start_time, feature, frequency, clust_class_list):
    result_df = pd.DataFrame()
    for clust_class in clust_class_list:
        for ms_name, class_value in clust_result.items():
            if class_value == str(clust_class):
                result_df = pd.concat([result_df, data[ms_name]])
    result_df.columns = [feature]
    time_index = pd.date_range(start=start_time, freq = str(frequency)+"T", periods=len(result_df))
    result_df.set_index(time_index, inplace = True)
    
    return result_df

def get_train_test_data_by_days(data, freq):
    data_day_length = int(len(data)/(24*60/freq))
    print("data_day_length : ", data_day_length)
    split_day_length = int(data_day_length*0.8)
    print("train_day_length : ", split_day_length)
    split_length = int(split_day_length*(24*60/freq))

    train = data[:split_length]
    test = data[split_length:]

    return train, test