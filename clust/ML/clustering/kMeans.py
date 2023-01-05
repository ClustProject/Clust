from Clust.clust.ML.clustering.clustering import Clustering 


from tqdm.autonotebook import tqdm 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
seed = 0
np.random.seed(seed)


parameter={
    "method":"kMeans", 
    "param":{}
}

class KKMeans():
    
    def __init__(self):
        pass
        
    # TODO
    def get_km_model(self, n_clusters, method, max_iter = 10):
        """
        get basic KMeans model by parameters
        
        Args:
            n_clusters(int): Number of clusters to form.
            method (string): {“euclidean”, “dtw”, “softdtw”} (default: “euclidean”)
                Metric to be used for both cluster assignment and barycenter computation. If “dtw”, DBA is used for barycenter computation.
            max_iter(int): Maximum number of iterations of the k-means algorithm for a single run. Default = 10
              
        Returns:
            km: (class tslearn.clustering.TimeSeriesKMeans)
            
        """
        from tslearn.clustering import TimeSeriesKMeans
        if method =='euclidean':
            km = TimeSeriesKMeans(n_clusters=n_clusters, 
                                    metric = method, 
                                    max_iter=max_iter, 
                                    verbose=True, 
                                    random_state=seed)

        elif method =='dtw':
            #max_iter_barycenter : int (default: 100), Number of iterations for the barycenter computation process. 
            #Only used if metric=”dtw” or metric=”softdtw”.
            km = TimeSeriesKMeans(n_clusters=n_clusters, 
                                    metric=method, 
                                    max_iter= max_iter, 
                                    max_iter_barycenter=max_iter, 
                                    verbose=True, 
                                    random_state=seed, 
                                    n_init=2)
        elif method == 'softdtw':
            km = TimeSeriesKMeans(n_clusters=n_clusters, 
                                    metric=method, 
                                    max_iter = max_iter, 
                                    verbose=True, 
                                    random_state=seed, 
                                    metric_params={"gamma": .01})
        else:
            km = TimeSeriesKMeans(n_clusters=n_clusters, max_iter=max_iter, verbose=True, random_state=seed)
        
        return km
            
        
    def show_clustering_result(self, n_clusters, X, y, method, model):
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
        print(custom_xlim, custom_ylim)
        fig, ax = plt.subplots(1, n_clusters, figsize=(20, 3))
        fig.suptitle("k_means:" + method, y=1.08)
        plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
        for yi in range(n_clusters):
            ax[yi].set_title('Clust '+str(yi))
            for xx in X[y == yi]:
                ax[yi].plot(xx.ravel(), "k-", alpha=.2)
            ax[yi].plot(model.cluster_centers_[yi].ravel(), "r-")
        plt.show()
        

    def get_oneCluster_result(self, data, n_clusters, method):
        """
        - get one cluster result. 
            1) make model 2) get cluster_label 3) show result by label
        
        Args:
            data (numpy.ndarray): data to be clustered
            n_clusters(int): number of clusters to form.
            method (string): k-means method {“euclidean”, “dtw”, “softdtw”} (default: “euclidean”)
            
        Returns:
            model: (class tslearn.clustering.TimeSeriesKMeans) :kmeans Model
            cluster_labels (numpy.ndarray): cluster_labels (numpy.ndarray): clustering result

        """

        model = self.get_km_model(n_clusters, method)
        cluster_labels = model.fit_predict(data)
        self.show_clustering_result(n_clusters, data, cluster_labels, method, model)

        return model, cluster_labels

    def get_multipleCluster_result(self, data, max_clusters, method):
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
        
        from sklearn.metrics import silhouette_score
        silhouette = []
        clusters_range = range(2, max_clusters)
        for n_clusters in tqdm(clusters_range):
            model, cluster_labels = self.get_oneCluster_result(data, n_clusters, method)
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette.append([n_clusters, silhouette_avg, model.inertia_])
        metric = pd.DataFrame(silhouette, columns=['n_clusters', "silhouette_score", "distortion_score"])
        metric = metric.set_index('n_clusters')
        return cluster_labels, metric