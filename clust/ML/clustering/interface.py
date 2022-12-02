from sklearn.cluster import DBSCAN
import pandas as pd
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

def clusteringByMethod(feature_dataset: [pd.DataFrame], feature_datasetName:[str], model:int, x:int=None, y:int=None):
    """ 
    make clustering result of multiple dataset bu clustering model name

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        feature_dataset: list of multiple dataframe inputs to be clustered
        feature_datasetName: list of multiple data frame name
        model: clust model name to be applied modelList = ['som']
        x: x length
        y: y length

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

    if (len(feature_datasetName)>0):
        if model =="som":
            from Clust.clust.ML.clustering.somClustering import SomClustering
            somV= SomClustering(feature_dataset, feature_datasetName, x, y)
            result = somV.train()
            figdata, figdata2= somV.make_figs()
    
    return result, figdata, figdata2