
from sklearn.cluster import DBSCAN
import pandas as pd

from Clust.clust.ML.clustering.somClustering import SomClustering
from Clust.clust.ML.clustering import plot_seriesDataSet
def ClusteringByMinPoints(data, minPts=3, method = "DBSCAN"):
    """ 
    
    Clustering

    :param data: input data
    :type data: DataFrame 

    :param minPts: minimum points for clusting
    :type minPts: integer

    :param method: Clustering Method
    :type method: string ["DBSCAN"]

    :return: result : clustering result
    :rtype: array of integer 

    example
        >>> inputData = data
        >>> minPts = 10
        >>> result = pd.DataFrame() 
        >>> result['predict'] = ClusteringByMinPoints.Clustering(inputData, minPts, "DBSCAN")
    """

    if method =="DBSCAN":
        model = DBSCAN(min_samples= minPts)
        result=model.fit_predict(data)
    return result

def clusteringByMethod(feature_dataset: [pd.DataFrame], feature_datasetName:[str], model:int, x:int=None, y:int=None):
    
    """ make clustering result of multiple dataset bu clustering model name

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        feature_dataset: list of multiple dataframe inputs to be clustered
        feature_datasetName: list of multiple data frame name
        model: clust model name to be applied modelList = ['som']
        x: x length
        y: y length

    Returns:
        A dict mapping keys to the corresponding clustering result number. 
        example:
        {b'ICW0W2000011': '5',
         b'ICW0W2000013': '4',
         b'ICW0W2000014': '6'...
        }
    """

    result =None
    figdata=None
    figdata2=None

    if (len(feature_datasetName)>0):
        if model =="som":
            from Clust.clust.ML.clustering.somClustering import SomClustering
            somV= SomClustering(feature_dataset, feature_datasetName, x, y)
            result = somV.train()
            figdata, figdata2= somV.make_figs()
    
    return result, figdata, figdata2


def get_somResultSet_by_features(series_data_set, series_data_set_name, xnum, ynum):
    fig_width = 30
    fig_height = 30
    fig_width_num = 4
    """
    get som result by all features, and show result graph
    Args:
        series_data_set(dict) :input series dataset, keys: feature_name, values: series data of a specific feature
        series_data_set_name(dict) : series dataset name:keys feaure_name, values: name of a series data
        xnum(int): xnum of som clustering
        ynum(int): ynum of som clustering
        
    Returns:
        resultSet(pd.DataFrame):result-> index: bucket_name, columns: feature_name_list, value: clustering number of each bucket based on feature  
    """
    resultSet = pd.DataFrame()
    feature_list = list(series_data_set.keys())
    for feature_name in feature_list:
        print(feature_name)
        feature_dataset= series_data_set[feature_name]
        feature_datasetName = series_data_set_name[feature_name]
        plt = plot_seriesDataSet.groupSeriesFig(fig_width, fig_height, feature_dataset, feature_datasetName, feature_name, fig_width_num )
        feature_dataset= series_data_set[feature_name]
        feature_datasetName = series_data_set_name[feature_name]
        somV= SomClustering(feature_dataset, feature_datasetName, xnum, ynum)
        result = somV.train()
        resultSet[feature_name] = pd.DataFrame.from_dict(result, orient='index')
        figdata, figdata2= somV.make_figs()
        plt.show()
    return resultSet

    