def clusteringByMethod(data, parameter):
    """ make clustering result of multiple dataset 

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

    from Clust.clust.tool.plot.util import plt_to_image
    from Clust.clust.ML.clustering.som import Som
    result = None
    figdata = None
    param = parameter['param']
    model_name = parameter['method']

    if (len(data.columns) > 0):
        if model_name == "som":
            som_i = Som(param)   
            data_series = som_i.make_input_data(data)
            data_name = list (data.columns)
            som_model = som_i.train(data_series)
                    
            """
            file_name = "model.pkl"
            #important code for external usage
            som_i.save_model(file_name)
            model = som_i.load_model(file_name)
            
            som_c = Som(param) 
            som_c.set_model(model)
            
            """
            som_c = som_i
            win_map = som_c.get_win_map(data_series)
            result = som_c.get_clustering_result(data_series)
            result_dic = som_c.get_result_dic(data_name, result)
            
            plt1 = som_c.plot_ts_by_label()
            plt1.show()
            figdata = plt_to_image(plt1)
            
            plt2 = som_c.plot_label_histogram()
            plt2.show()
        
        return result_dic, figdata

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

