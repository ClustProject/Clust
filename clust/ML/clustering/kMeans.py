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
class KMeans(Clustering):   
    def __init__(self, param):
        
        """
        Args:
        param(dict): parameter for clustering
            >>> param = {"som_x":2,
                             "som_y":2,
                             "neighborhood_function":"gaussian",
                             "activation_distance":"euclidean",
                             "epochs":5000}
        """
        super().__init__(param)
        
        
    def interpret_param(self, param):
        """interpret_clustering parameter, overriding from super class
        
        """
        self.som_x = param.get('som_x')


    def train(self, data):
        """ train miniSom amd return miniSom instance
        Args:
            data(series):input data
        
        Return:
            som (MiniSom instnace): MiniSom result instance
        """
        data_length = (data.shape[1])

        result = {}
        self.som = MiniSom(self.som_x, self.som_y, data_length, sigma=0.3, learning_rate = 0.1)
        self.som.random_weights_init(data)
        self.som.train(data, self.epochs)
  
        return self.som

    
    def plot_ts_by_label(self):
        """ overriding
        """
        som_x = self.som_x
        som_y = self.som_y
        win_map = self.win_map
        center_type = 'dtw_barycenter_averaging'
    
        plt.rcParams.update({'font.size': 25})
        if(len(win_map)==1):
            fig = plt.figure(figsize=(25,25))
            axs = fig.add_subplot(1,1,1)
            cluster = (0,0)
            if cluster in win_map.keys():
                for series in win_map[cluster]:
                    axs.plot(series,c="gray",alpha=0.5)
                # nan to zero for plotting
                """
                for m in win_map[cluster]:
                    m = np.nan_to_num(m, copy=False)
                """
                if center_type == 'dtw_barycenter_averaging':
                    axs.plot(dtw_barycenter_averaging(np.vstack(win_map[cluster])),c="red") 
                else:
                    axs.plot(np.average(np.vstack(win_map[cluster]),axis=0),c="red")
                
                axs.set_title(f"Cluster {1}")
        else:
            size_x = math.ceil((math.sqrt(len(win_map))))
            size_y = math.ceil(len(win_map)/size_x)
            fig, axs = plt.subplots(size_y,size_x,figsize=(25,25))
            cnt = 0
            for x in range(som_x):
                for y in range(som_y):
                    cluster = (x,y)
                    if cluster in win_map.keys():
                        if(size_y==1): pos = cnt
                        else : pos = (int(cnt/size_x),cnt%size_x)
                        cnt = cnt + 1
                        for series in win_map[cluster]:
                            axs[pos].plot(series,c="gray",alpha=0.5)
                        
                        if center_type == 'dtw_barycenter_averaging':
                            axs[pos].plot(dtw_barycenter_averaging(np.vstack(win_map[cluster])),c="red") 
                        else:
                            axs[pos].plot(np.average(np.vstack(win_map[cluster]),axis=0),c="red")
                        cluster_number = x*som_y+y+1
                        axs[pos].set_title(f"Cluster {cluster_number}")

        fig.suptitle('Clusters')
        
        return plt
    
    """
    New Module
    """
    
    def plot_label_histogram(self):
        """ overriding
        """
        som_x = self.som_x
        som_y = self.som_y
        win_map = self.win_map
        
        cluster_c = []
        cluster_n = []
        for x in range(som_x):
            for y in range(som_y):
                cluster = (x,y)
                if cluster in win_map.keys():
                    cluster_c.append(len(win_map[cluster]))
                else:
                    cluster_c.append(0)
                cluster_number = x*som_y+y+1
                cluster_n.append(f"Cluster {cluster_number}")
        plt.title("Cluster Distribution for SOM")
        plt.bar(cluster_n,cluster_c)

        return plt

    
    def get_win_map(self, data):
        """ Returns the mapping of the winner nodes and inputs
        Args:
            data(series):input data
        
        Return:
            som (MiniSom instnace): MiniSom result instance
        """
        self.win_map = self.som.win_map(data)
        return self.win_map
    
    def get_clustering_result(self, data):
        
        """get clustering number result
        Args:
            data(series):data
            
        Return:
            cluster_map(array): cluster map result of input data
            >>> example> [1, 2, 0]
        """
        som_y = self.som_y
        som = self.som
        # return dataframe
        cluster_map = []
        
        for idx in range(len(data)):
            winner_node = som.winner(data[idx])
            cluster_map.append(str(winner_node[0]*som_y+winner_node[1]+1))

        return cluster_map



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