from tqdm.autonotebook import tqdm 
import numpy as np
import pandas as pd
from Clust.clust.ML.clustering.clustering import Train, Test
from tslearn.clustering import TimeSeriesKMeans
seed = 0
np.random.seed(seed)

class KMeansTrain(Train):   
    def __init__(self):
        super().__init__()

    def set_param(self, param):
        """
        set param

        Args:
            param(dict): parameter for clustering

        >>> param = {"n_clusters":3,
                    "metric":"euclidean"}

        """
        self.n_clusters = param.get('n_clusters')
        self.metric = param.get('metric')

    def train(self, data):
        """ 
        fit K-Means

        Args:
            data(series):input data
        """
        seed = 0
        np.random.seed(seed)
        self.model = TimeSeriesKMeans(n_clusters=self.n_clusters, metric = self.metric, random_state=seed)
        self.model.fit(data)


class KMeansTest(Test):
    def __init__(self):
        super().__init__()

    def predict(self, data):
        """g
        et calustering label

        Args:
            data(series):data
            
        Return:
            self.y(array): label result
            
        >>> example> [1, 2, 0]
        """

        self.X = data

        # return dataframe
        #label = []
        
        self.y = self.model.predict(data)

        return self.y


def search_best_n_clust(data, param):
    """
    get multiple cluster result. (n_cluster = 2 ~ param['n_cluster])
    
    1) get cluster labels 
    2) make silhouette and distortion score matrics
    
    Args:
        data (numpy.ndarray): data to be clustered
        max_cluster_num(int): Max number of clusters to form.     
        
    Returns:
        metric (dataFrame): dataframe with silhouette_score and distortion_score

    """
    from sklearn.metrics import silhouette_score
    silhouette = []
    max_clusters = param['n_clusters']
    clusters_range = range(2, max_clusters)
    
    for n_clusters in tqdm(clusters_range):
        param['n_clusters'] = n_clusters
        clust_train = KMeansTrain()
        clust_train.set_param(param)
        clust_train.train(data)

        clust_test = KMeansTest() 
        clust_test.set_model(clust_train.model)
        
        result = clust_test.predict(data)
        plt = clust_test.plot_ts_by_label(data, result)
        plt.show()

        silhouette_avg = silhouette_score(data, result)
        silhouette.append([n_clusters, silhouette_avg, clust_train.model.inertia_])

    metric = pd.DataFrame(silhouette, columns=['n_clusters', "silhouette_score", "distortion_score"])
    metric = metric.set_index('n_clusters')
    
    return metric


    