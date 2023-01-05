import pickle
class Clustering:
    """Clustering Super Class"""

    def __init__(self):
        pass

    def get_dict_from_two_array(self, input_name, label_result):
        """make dictionary from two array (key array, value array)

        Args:
            input_name (array): input name of data
            label_result (array): clustering result 

        Returns:
            dict_result(dict): dictionary type result -> key: nput_name, value: label
        """
        dict_result = dict(zip(input_name, label_result))
        return dict_result

    # TODO overriding
    def make_input_data(self, data):
        """make input data for clustering. 

        """
        series_data = data.to_numpy().transpose()
        return series_data

class train:
    """Clustering Train Super Class"""

    def __init__(self, param):
        """ param interpretation

        Args:
            param (dict): parameter for clustering by specific clustering methods
        """
        self.interpret_param(param)

    def save_model(self, model_file_address):
        """ save model: dafult file type is pickle
        
        Args:
            model_file_address(str) : model_file_address
        
        """
        with open(model_file_address, 'wb') as outfile:
            pickle.dump(self.som, outfile)


    # TODO overriding
    def interpret_param(self, param):
        """ interpret_clustering parameter. Each method should define interpret_param module.
        
        """
        pass
    
    def train(self):
        """training model Each method should define

        """
        pass
    
    


class test:
    """Clustering Super Class"""

    def __init__(self):
        pass
    

    def load_model(self, model_file_address):
        """ load model: : dafult file type is pickle
        Args:
            model_file_address(str) : model_file_address
        
        """
        with open(model_file_address, 'rb') as infile:
            self.model = pickle.load(infile)
        return self.model
    
    def set_model(self, model):
        """ set new model

        Args:
            model
            
        """
        self.model = model

    ## TODO overriding
    def predict(self, data):
        """make winner_node (self.win_map) and get calustering label

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