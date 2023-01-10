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
    from Clust.clust.ML.clustering.som_YK import SomTrain, SomTest
    from Clust.clust.ML.clustering.kMeans_YK import KMeansTrain, KMeansTest

    result = None
    figdata = None
    param = parameter['param']
    model_name = parameter['method']

    if (len(data.columns) > 0):
        # Train/Test 클래스 생성
        # 모델 저장/로드 경로
        if model_name == "som":
            clust_train = SomTrain(param)
            clust_test = SomTest()
            save_path = "./som.pkl"
            load_path = "./som.pkl"
        elif model_name == "kmeans":
            clust_train = KMeansTrain(param)
            clust_test = KMeansTest()
            save_path = "./km.pkl"
            load_path = "./km.pkl"

        # 1. 데이터 준비
        data_series = clust_train.make_input_data(data)
        data_name = list(data.columns)

        # 2. train
        clust_train.train(data_series)

        # 3. model save
        clust_train.save_model(save_path)

        # 4. model load
        clust_test.load_model(load_path)

        # 5. test (predict)
        result = clust_test.predict(data_series)
        result_dic = clust_test.get_dict_from_two_array(data_name, result)

        # plot time series style
        plt1 = clust_test.plot_ts_by_label()
        plt1.show()
        figdata = plt_to_image(plt1)
            
        # plot historgram
        plt2 = clust_test.plot_label_histogram()
        plt2.show()
        
        return result_dic, figdata
