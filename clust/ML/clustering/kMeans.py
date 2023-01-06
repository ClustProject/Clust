from tqdm.autonotebook import tqdm 
import numpy as np
import pandas as pd
from Clust.clust.ML.clustering.clustering import Clustering, train, test
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
class KMeansTrain(train, Clustering):   
    def __init__(self, param):
        
        """
        Args:
        param(dict): parameter for clustering
            >>> param = {"n_clusters":3,
                        "metric":"euclidean"}
        """
        super().__init__(param)


    def _interpret_param(self, param):
        """interpret_clustering parameter, overriding from super class
        
        """
        self.n_clusters = param.get('n_clusters')
        self.metric = param.get('metric')

    def train(self, data):
        """ train miniSom amd return miniSom instance
        Args:
            data(series):input data
        
        Return:
            som (MiniSom instnace): MiniSom result instance
        """
        seed = 0
        np.random.seed(seed)
        self.model = TimeSeriesKMeans(n_clusters=self.n_clusters, metric = self.metric, random_state=seed)
        self.model.fit(data)

    # New Function
    def search_best_n_clust(self, data, max_clusters):
        """
        - get multiple cluster result. 
            1) get cluster labels 2) make silhouette and distortion score matrics
        
        Args:
            data (numpy.ndarray): data to be clustered
            max_clusters(int): Max number of clusters to form.
            method (string): k-means method {“euclidean”, “dtw”, “softdtw”} (default: “euclidean”)
            
        Returns:
            cluster_labels (numpy.ndarray): clustering result
            metric (dataFrame): dataframe with silhouette_score and distortion_score

        """
        seed = 0
        np.random.seed(seed)
        from sklearn.metrics import silhouette_score
        silhouette = []
        clusters_range = range(2, max_clusters)
        kMeeans_t = KMeansTest()
        for n_clusters in tqdm(clusters_range):
            model = TimeSeriesKMeans(n_clusters=n_clusters, metric = self.metric, random_state=seed)
            label = model.fit_predict(data)
            plt = kMeeans_t.plot_ts_by_label(n_clusters, data, label, self.metric, model)
            plt.show()
            silhouette_avg = silhouette_score(data, label)
            silhouette.append([n_clusters, silhouette_avg, model.inertia_])
        metric = pd.DataFrame(silhouette, columns=['n_clusters', "silhouette_score", "distortion_score"])
        metric = metric.set_index('n_clusters')
        return metric


class KMeansTest(test, Clustering):   
    def predict(self, data):
        """get calustering label

        Args:
            data(series):data
            
        Return:
            cluster_map(array): cluster map result of input data
            >>> example> [1, 2, 0]
        """

        # return dataframe
        #label = []
        
        label = self.model.predict(data)

        return label

    def plot_ts_by_label(self, n_clusters, X, y, method, model):
        """
        Show clustering result 
        
        Args:
            n_clusters (int):Number of clusters to form.
            X (numpy.ndarray): 2d array of timeseries dataset
            y (numpy.ndarray): 1d array (label result)
            method (string): k-means method {“euclidean”, “dtw”, “softdtw”} (default: “euclidean”)
            model: (class tslearn.clustering.TimeSeriesKMeans) :kmeans Model

        """
        custom_xlim = [0, X.shape[1]]
        custom_ylim = [X.min(), X.max()]
        fig, ax = plt.subplots(1, n_clusters, figsize=(20, 3))
        fig.suptitle("k_means:" + method, y=1.08)
        plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
        for yi in range(n_clusters):
            ax[yi].set_title('Clust '+str(yi))
            for xx in X[y == yi]:
                ax[yi].plot(xx.ravel(), "k-", alpha=.2)
            ax[yi].plot(model.cluster_centers_[yi].ravel(), "r-")

        return plt

    def plot_label_histogram(self, som_x, som_y):
        pass