import matplotlib.pyplot as plt
import numpy as np
# from Clust.clust.ML.clustering.clustering import Clustering, train, test
from Clust.clust.ML.clustering.clustering_YK import Clustering, Train, Test
from minisom import MiniSom 
from tslearn.barycenters import dtw_barycenter_averaging
import math
import pickle
#plt.switch_backend('Agg')


class SomTrain(Clustering, Train):   
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
        super().__init__()
        self._interpret_param(param)
        
    def _interpret_param(self, param):
        """interpret_clustering parameter, overriding from super class
        
        """
        self.som_x = param.get('som_x')
        self.som_y = param.get('som_y')
        self.epochs = param.get('epochs')
        self.neighborhood_function  = param.get("neighborhood_function")
        self.activation_distance = param.get("activation_distance")
        """
        if self.som_x is None:
            self.som_x = self.som_y = math.ceil(math.sqrt(math.sqrt(len(self.data_series))))
        """

    def train(self, data):
        """ train miniSom amd return miniSom instance
        Args:
            data(series):input data
        
        Return:
            som (MiniSom instnace): MiniSom result instance
        """
        data_length = (data.shape[1])

        result = {}
        self.model = MiniSom(self.som_x, self.som_y, data_length, sigma=0.3, learning_rate = 0.1)
        self.model.random_weights_init(data)
        self.model.train(data, self.epochs)

    def save_model(self, model_file_path):
        """ overriding
        """
        with open(model_file_path, 'wb') as outfile:
            pickle.dump(self.model, outfile)


class SomTest(Clustering, Test):
    def __init__(self):
        super().__init__()

    def load_model(self, model_file_path):
        """ overriding
        """
        # load model
        with open(model_file_path, 'rb') as infile:
            model = pickle.load(infile)

        # set model
        self.set_model(model)

    def predict(self, data):
        """make winner_node (self.win_map) and get clustering label

        Args:
            data(series):data
            
        Return:
            cluster_map(array): cluster map result of input data
            >>> example> [1, 2, 0]
        """
        self.win_map = self.model.win_map(data)

        # return dataframe
        label = []
        
        for idx in range(len(data)):
            winner_node = self.model.winner(data[idx])
            # label.append(str(winner_node[0]*som_y+winner_node[1]+1))
            label.append(str(winner_node[0]*len(self.model._neigy)+winner_node[1]+1))

        return label

    # TODO plot이 win_map에 의해 그려지도록 되어있지만, 되도록 data (series) 와 Result(array) 에 의해서 그려지도록 하는게 맞음
    # input을 data_series, label_result로 그리도록 (self.win_map 사용해도 되긴함)

    def plot_ts_by_label(self):
        """ overriding
        """
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
            for x in range(len(self.model._neigx)):
                for y in range(len(self.model._neigy)):
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
                        cluster_number = x*len(self.model._neigy)+y+1
                        axs[pos].set_title(f"Cluster {cluster_number}")

        fig.suptitle('Clusters')
        
        return plt

    # TODO plot이 win_map에 의해 그려지도록 되어있지만, 되도록 data와 Result에 의해서 그려지도록 하는게 맞음
    # input을 data_series, label_result로 그리도록 (self.win_map 사용해도 되긴함)

    def plot_label_histogram(self):
        """ overriding
        """
        win_map = self.win_map
        
        cluster_c = []
        cluster_n = []
        for x in range(len(self.model._neigx)):
            for y in range(len(self.model._neigy)):
                cluster = (x,y)
                if cluster in win_map.keys():
                    cluster_c.append(len(win_map[cluster]))
                else:
                    cluster_c.append(0)
                cluster_number = x*len(self.model._neigy)+y+1
                cluster_n.append(f"Cluster {cluster_number}")
        plt.title("Cluster Distribution for SOM")
        plt.bar(cluster_n,cluster_c)

        return plt