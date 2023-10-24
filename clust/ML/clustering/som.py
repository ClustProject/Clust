import matplotlib.pyplot as plt
import numpy as np
from Clust.clust.ML.clustering.clustering import Train, Test
from minisom import MiniSom 

#plt.switch_backend('Agg')
class SomTrain(Train):   
    def __init__(self):
        
        """
        Args:
            param(dict): parameter for clustering

        >>> param = {"som_x":2,
        ...                 "som_y":2,
        ...                 "neighborhood_function":"gaussian",
        ...                 "activation_distance":"euclidean",
        ...                 "epochs":5000}
        """
        super().__init__()
        
    def set_param(self, param):
        """
        interpret_clustering parameter, overriding from super class
        
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
        """ 
        train miniSom amd return miniSom instance

        Args:
            data(series):input data
        
        Return:
            som (MiniSom instnace): MiniSom result instance
        """
        data_length = (data.shape[1])

        self.model = MiniSom(self.som_x, self.som_y, data_length, sigma=0.3, learning_rate = 0.1)
        self.model.random_weights_init(data)
        self.model.train(data, self.epochs)


class SomTest(Test):
    def __init__(self):
        super().__init__()

    def predict(self, data):
        """
        make winner_node (self.win_map) and get clustering label

        Args:
            data(series):data
            
        Return:
            cluster_map(array): cluster map result of input data

        >>> example> [1, 2, 0]
        
        """
        self.X = data
        self.win_map = self.model.win_map(data)

        # return dataframe
        self.y  = []
        
        
        for idx in range(len(data)):
            winner_node = self.model.winner(data[idx])
            self.y.append(winner_node[0]*len(self.model._neigy)+winner_node[1])
        
        # self.X, self.y, self.cluster_centers_
        self.y = np.array(self.y)
        
        
        return self.y
    