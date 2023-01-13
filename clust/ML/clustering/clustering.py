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
            label(array): label result by
            >>> example> [1, 2, 0]
        """
        
        label =[]

        return label

    def plot_ts_by_label(self):
        """ plot timeseries style result with representative clustering value.
        
        """
        pass
    

    def plot_label_histogram(self):
        """ plot histogram result with clustered result

        """
        pass