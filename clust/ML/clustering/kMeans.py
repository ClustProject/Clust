from tqdm.autonotebook import tqdm 
import numpy as np
import pandas as pd
from Clust.clust.ML.clustering.clustering import Train, Test
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
seed = 0
np.random.seed(seed)

class KMeansTrain(Train):   
    def __init__(self):
        super().__init__()

    def set_param(self, param):
        """
        Args:
        param(dict): parameter for clustering
            >>> param = {"n_clusters":3,
                        "metric":"euclidean"}
        """
        self.n_clusters = param.get('n_clusters')
        self.metric = param.get('metric')

    def train(self, data):
        """ fit K-Means
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
        """get calustering label

        Args:
            data(series):data
            
        Return:
            cluster_map(array): cluster map result of input data
            >>> example> [1, 2, 0]
        """

        self.X = data

        # return dataframe
        #label = []
        
        label = self.model.predict(data)
        self.y = label

        return label

    def plot_ts_by_label(self):
        """
        Show clustering result 
        
        Args:
            n_clusters (int):Number of clusters to form.
            X (numpy.ndarray): 2d array of timeseries dataset
            y (numpy.ndarray): 1d array (label result)
            method (string): k-means method {“euclidean”, “dtw”, “softdtw”} (default: “euclidean”)
            model: (class tslearn.clustering.TimeSeriesKMeans) :kmeans Model

        """
        n_clusters = self.model.cluster_centers_.shape[0]

        custom_xlim = [0, self.X.shape[1]]
        custom_ylim = [self.X.min(), self.X.max()]
        fig, ax = plt.subplots(1, n_clusters, figsize=(20, 3))
        fig.suptitle("k_means:" + self.model.metric, y=1.08)
        plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
        for yi in range(n_clusters):
            ax[yi].set_title('Clust '+str(yi+1))
            for xx in self.X[self.y == yi]:
                ax[yi].plot(xx.ravel(), "k-", alpha=.2)
            ax[yi].plot(self.model.cluster_centers_[yi].ravel(), "r-")

        return plt

    def plot_label_histogram(self):
        """ overriding
        """
        n_clusters = self.model.cluster_centers_.shape[0]
        label = self.y

        cluster_c = [0 for i in range(n_clusters)]
        cluster_n = [f"Cluster {i+1}" for i in range(n_clusters)]

        for i in range(len(label)):
            cluster_c[label[i]] += 1
        
        plt.title("Cluster Distribution for K-Means")
        plt.bar(cluster_n, cluster_c)

        return plt


def search_best_n_clust(data, param):
    """
    - get multiple cluster result. (n_cluster = 2 ~ param['n_cluster])
        1) get cluster labels 2) make silhouette and distortion score matrics
    
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
        plt = clust_test.plot_ts_by_label()
        plt.show()

        silhouette_avg = silhouette_score(data, result)
        silhouette.append([n_clusters, silhouette_avg, clust_train.model.inertia_])

    metric = pd.DataFrame(silhouette, columns=['n_clusters', "silhouette_score", "distortion_score"])
    metric = metric.set_index('n_clusters')
    
    return metric


    