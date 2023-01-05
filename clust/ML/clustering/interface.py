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
    from Clust.clust.ML.clustering.som import SomTrain, SomTest
    result = None
    figdata = None
    param = parameter['param']
    model_name = parameter['method']

    if (len(data.columns) > 0):
        if model_name == "som":
            som_i = SomTrain(param) 
            # 1. 데이터 준비
            data_series = som_i.make_input_data(data)
            data_name = list(data.columns)
            # 2. Train
            som_i.train(data_series)
            model = som_i.model
            """
            #important code for external usage
            # 3. model save 
            file_name = "model.pkl"
            SomTrain.save_model(file_name)
            """

            #5. Test
            """
            #important code for external usage
            #4. model Load
            file_name = "model.pkl"
            model = som_t.load_model(file_name)
            """
            som_t = SomTest()
            som_t.set_model(model)

            # TODO Hard Coding 삭제해야함 (som_x, som_y 를 인풋으로 넣으면 안됨)
            som_x = 2
            som_y = 2
            result = som_t.predict(data_series, som_y) 
            result_dic = som_t.get_dict_from_two_array(data_name, result)
            
            plt1 = som_t.plot_ts_by_label(som_x, som_y)
            plt1.show()
            figdata = plt_to_image(plt1)
            
            plt2 = som_t.plot_label_histogram(som_x, som_y)
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

