from Clust.clust.ML.clustering.som import SomTrain, SomTest
from Clust.clust.ML.clustering.kMeans import KMeansTrain, KMeansTest
from Clust.clust.ML.tool.data import DF_to_series
from Clust.clust.ML.tool.model import load_pickle_model, save_pickle_model

#TODO 나중에 수정해야함 전반적인 구조들과 스트럭쳐

def clusteringByMethod(data, parameter, model_path):
    """ make clustering result of multiple dataset 

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        data (dataFrame): list of multiple dataframe inputs to be clustered (each column : individual data, index : time sequence data)
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

    result = None
    param = parameter['param']
    model_name = parameter['method']

    if (len(data.columns) > 0):
        # Train/Test 클래스 생성
        # 모델 저장/로드 경로
        if model_name == "som":
            clust_train = SomTrain()
            clust_train.set_param(param)
            clust_test = SomTest()
        elif model_name == "kmeans":
            clust_train = KMeansTrain()
            clust_train.set_param(param)
            clust_test = KMeansTest()
        print("start..........................")
        # 1. 데이터 준비
        data_series = DF_to_series(data)
        

        # 2. Train
        clust_train.set_param(param)
        clust_train.train(data_series)

        # 3. model save
        save_pickle_model(clust_train.model, model_path)

        # 4. model load
        model = load_pickle_model(model_path)
        
        # 5. predict
        clust_test.set_model(model)
        result = clust_test.predict(data_series)

        # 6. test 
        """
        from sklearn.metrics import calinski_harabasz_score
        print(data_series, result)
        score = calinski_harabasz_score(data_series, result)
        print("Score:", score)
        """
        # 7. test data plot
        import matplotlib.pyplot as plt
        
        plt.rcParams['figure.figsize'] =(18, 5)
        plt1 = clust_test.plot_ts_by_label(data_series, result)
        
    
        
        return result, plt1



