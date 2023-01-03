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

