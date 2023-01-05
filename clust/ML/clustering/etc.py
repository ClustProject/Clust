from sklearn.cluster import DBSCAN

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

    if method == "DBSCAN":
        model = DBSCAN(min_samples = minPts)
        result = model.fit_predict(data)
    return result

