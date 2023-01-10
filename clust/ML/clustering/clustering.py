import pickle
class Clustering:
    """Clustering Super Class"""

    def __init__(self):
        pass

    def get_dict_from_two_array(self, input_key, input_value):
        """make dictionary from two array (key array, value array)

        Args:
            input_key (array): input array for key
            input_value (array): input array for value

        Returns:
            dict_result(dict): dictionary type result -> key: input_key, value: input_value
        """
        dict_result = dict(zip(input_key, input_value))
        return dict_result

    # TODO overriding
    def make_input_data(self, data):
        """make input data for clustering. 
        Args:
            data(np.dataFrame): input data
        Return:
            series_data(series): transformed data for training, and prediction

        """
        series_data = data.to_numpy().transpose()
        return series_data


class Train:
    """Clustering Train Super Class"""

    def __init__(self, param):
        """ param interpretation

        Args:
            param (dict): parameter for clustering by specific clustering methods
        """  
        self._interpret_param(param)

    def save_model(self, model_file_path):
        """ save model: dafult file type is pickle
        
        Args:
            model_file_path(str) : model_file_path
        
        """
        with open(model_file_path, 'wb') as outfile:
            pickle.dump(self.model, outfile)


    # TODO overriding
    def _interpret_param(self, param):
        """ interpret_clustering parameter. Each method should define interpret_param module.
        
        """
        pass
    
    def train(self):
        """training model Each method should define

        """
        pass
    

class Test:
    """Clustering Super Class"""

    def __init__(self):
        pass
    

    def load_model(self, model_file_path):
        """ load model: : dafult file type is pickle
        Args:
            model_file_path(str) : model_file_path
        
        """
        with open(model_file_path, 'rb') as infile:
            model = pickle.load(infile)
        
        self.set_model(model)
    
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
    

    def plot_label_histogram(self, label):
        """ plot histogram result with clustered result
        
        Args:
            label(array): label result by

        """
        pass