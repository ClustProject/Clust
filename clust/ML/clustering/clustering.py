import matplotlib.pyplot as plt
import math
class Train:
    """Clustering Train Super Class"""

    def __init__(self):
        """ 
        param interpretation

        Args:
            param (dict): parameter for clustering by specific clustering methods
        """  
        pass

    # TODO overriding
    def set_param(self, param):
        """ 
        param interpretation

        Args:
            param (dict): parameter for clustering by specific clustering methods
        """  
        pass
    
    def train(self, data):
        """
        training model Each method should define

        Args:
            data (series): input data for training
        """
        
        pass

class Test:
    """

    Clustering Super Class

    """

    def __init__(self):
        pass
    
    def set_model(self, model):
        """ 
        set new model

        Args:
            model
            
        """
        self.model = model

    ## TODO overriding
    def predict(self, data):
        """
        get calustering label

        Args:
            data(series):data
            
        Return:
            self.y(array): label result by

        >>> example> [1, 2, 0]
        """
        self.X = data
        self.y = []

        return self.y

    def plot_ts_by_label(self, X, y):
        """
        Show clustering result 
            
        Args:
            X (numpy.ndarray): 2d array of timeseries dataset
            y (numpy.ndarray): 1d array (label result)    
            cluster_centers_ (numpy.ndarray): 1d array 
        """
        
        def get_cluster_centers(center_type):
            cluster_centers_ = []
            n_clusters = max(y)+1
            for i in range(0, n_clusters):
                if i in y:
                    cluster_result = X[y == i]
                    if center_type == 'dtw_barycenter_averaging':
                        from tslearn.barycenters import dtw_barycenter_averaging
                        cluster_centers_[i] = dtw_barycenter_averaging(cluster_result)
                    else:
                        cluster_centers_.append(cluster_result.mean(axis=0)) 
                else:
                    cluster_centers_.append([]) 
            return cluster_centers_
    
        cluster_centers_ = get_cluster_centers('mean')

        class_list = list(set(y))
        n_clusters = max(class_list)+1
        custom_xlim = [0, X.shape[1]]
        custom_ylim = [0, X.max()]
        
        col_num = 2
        row_num = math.ceil(n_clusters/col_num)
        fig, ax = plt.subplots(row_num, col_num, figsize=(20, row_num * 5))
        plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
        for i in range(0, row_num):
            for j in range(0, col_num):
                clust_num = (col_num*i+j)
                if clust_num+1 <= n_clusters:
                    ax[i][j].set_title('Clust '+str(clust_num))
                    for xx in X[y == clust_num]:
                        ax[i][j].plot(xx.ravel(), "k-", alpha=.2)
                    if len(cluster_centers_) > clust_num: #
                        ax[i][j].plot(cluster_centers_[clust_num].ravel(), "r-")
            
        return plt
    
    
    def select_specific_label_df(self, label, df, y):
        """
        select only specific column data of dataframe based on label

        Args:
            label (int): specific class name
            df (pd.DataFrame): input dataframe (each column data is input)
            y (numpy.ndarray): 1d array (label result)   

        Returns:
            dataframe: final_df(final selected df)
        """
        column_list=list(df.columns)

        label_column = []
        for i, result_class  in enumerate(y):
            if label == result_class:
                column_name = column_list[i]
                label_column.append(column_name)
        final_df = df[label_column]
        column_list=list(final_df.columns)

        return final_df 
    
    