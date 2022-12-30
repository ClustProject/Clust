from sklearn.cluster import DBSCAN
import pandas as pd

from Clust.clust.ML.clustering.somClustering import SomClustering
from Clust.clust.ML.clustering import plot_seriesDataSet

def clusteringByMethod(data, model, x=None, y=None):
    """ 
    make clustering result of multiple dataset 

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        data (dataFrame): list of multiple dataframe inputs to be clustered
        model(int): clust model name to be applied modelList = ['som']
        x(int): x length
        y(int): y length

    Returns:
        Dictionary: result (A dict mapping keys to the corresponding clustering result number)
        String: figdata (image)
        String: figdata2 (image)
    
    **Return Result Example**::

        result = { b'ICW0W2000011': '5',
                   b'ICW0W2000013': '4',
                   b'ICW0W2000014': '6'... }
    """

    result =None
    figdata=None
    figdata2=None
    if (len(data.columns)>0):
        if model =="som":
            data_series = data.to_numpy().transpose()
            data_name = list(data.columns)
            somV= SomClustering(data_series, data_name, 2, 2)
            result = somV.train()
            figdata, figdata2= somV.make_figs()
    
    return result, figdata, figdata2


def ClusteringByMinPoints(data, minPts=3, method = "DBSCAN"):
    """ 
    Clustering

    Args:
        data (DataFrame): input data
        minPts (integer): minimum points for clusting(3)
        method (string): Clustering Method(["DBSCAN"])

    Returns:
        array of integer: clustering result

    Example:
        >>> inputData = data
        >>> minPts = 10
        >>> result = pd.DataFrame() 
        >>> result['predict'] = ClusteringByMinPoints.Clustering(inputData, minPts, "DBSCAN")

    """

    if method =="DBSCAN":
        model = DBSCAN(min_samples= minPts)
        result=model.fit_predict(data)
    return result



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

    