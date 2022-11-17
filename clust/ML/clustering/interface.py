
from sklearn.cluster import DBSCAN
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

def clusteringByMethod(feature_dataset, feature_datasetName, model):
    result =None
    figdata=None
    figdata2=None
    if (len(feature_datasetName)>0):
        if model =="som":
            from clust.ML.clustering import som_visual
            result, figdata, figdata2 = som_visual.somTrain(feature_dataset, feature_datasetName)
    
    return result, figdata, figdata2