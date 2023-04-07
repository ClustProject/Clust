import matplotlib.pyplot as plt
import math
import numpy as np
class Train:
    """Clustering Train Super Class"""

    def __init__(self):
        """ param interpretation

        Args:
            param (dict): parameter for clustering by specific clustering methods
        """  
        pass

    # TODO overriding
    def set_param(self, param):
        """ param interpretation

        Args:
            param (dict): parameter for clustering by specific clustering methods
        """  
        pass
    
    def train(self, data):
        """training model Each method should define
        Args:
            data (series): input data for training
        """
        
        pass

class Test:
    """Clustering Super Class"""

    def __init__(self):
        pass
    
    def set_model(self, model):
        """ set new model

        Args:
            model
            
        """
        self.model = model

    ## TODO overriding
    def predict(self, data):
        """get calustering label

        Args:
            data(series):data
            
        Return:
            self.y(array): label result by
            >>> example> [1, 2, 0]
        """
        self.X = data
        self.y = []

        return self.y

    def plot_ts_by_label(self):
        """
            Show clustering result 
            
            Args:
                X (numpy.ndarray): 2d array of timeseries dataset
                y (numpy.ndarray): 1d array (label result)    
                cluster_centers_ (numpy.ndarray): 1d array 
        """
        X = self.X
        y = self.y
        
        def get_cluster_centers(center_type):
            cluster_centers_ = []
            self.n_clusters = max(self.y)+1
            for i in range(0, self.n_clusters):
                cluster_result = self.X[self.y == i]
                if center_type == 'dtw_barycenter_averaging':
                    from tslearn.barycenters import dtw_barycenter_averaging
                    cluster_centers_[i] = dtw_barycenter_averaging(cluster_result)
                else:
                    cluster_centers_.append(cluster_result.mean(axis=0)) 
            return cluster_centers_
    
        cluster_centers_ = get_cluster_centers('mean')
        n_clusters = y.max()
        custom_xlim = [0, X.shape[1]]
        custom_ylim = [0, X.max()]
        col_num = 2
        row_num = math.ceil(n_clusters/col_num)
        fig, ax = plt.subplots(col_num, row_num, figsize=(20, row_num * 5))
        plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
        for i in range(0, row_num):
            for j in range(0, col_num):
                clust_num = (col_num*i+j)
                ax[i][j].set_title('Clust '+str(clust_num))
                for xx in X[y == clust_num]:
                    ax[i][j].plot(xx.ravel(), "k-", alpha=.2)
                ax[i][j].plot(cluster_centers_[clust_num].ravel(), "r-")
            
        return plt
    
    def plot_label_histogram(self):
        bins = np.arange(0, self.y.max()+1.5)-0.5
        fig, ax = plt.subplots()
        _ = ax.hist(self.y, bins)
        ax.set_xticks(bins+0.5)

        return plt