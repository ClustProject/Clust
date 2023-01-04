import pickle

class Clustering:
    """Clustering Super Class"""

    def __init__(self, param):
        """ param interpretation

        Args:
            param (dict): parameter for clustering by specific clustering methods
        """
        self.interpret_param(param)
        
    def interpret_param(self, param):
        """ interpret_clustering parameter. Each method should define interpret_param module.
        
        """
        pass
    
    def make_input_data(self, data):
        """make input data for clustering. 

        """
        series_data = data.to_numpy().transpose()
        return series_data
    
    def set_model(self, model):
        """ set new model

        Args:
            model
            
        """

        self.som = model
        
    def save_model(self, model_file_address):
        """ save model: dafult file type is pickle
        
        Args:
            model_file_address(str) : model_file_address
        
        """
        with open(model_file_address, 'wb') as outfile:
            pickle.dump(self.som, outfile)

    def load_model(self, model_file_address):
        """ load model: : dafult file type is pickle
        Args:
            model_file_address(str) : model_file_address
        
        """
        with open(model_file_address, 'rb') as infile:
            self.som = pickle.load(infile)
        return self.som
    

    def train(self):
        """training model Each method should define

        """
        pass
    

    def get_result_dic(self, input_name, label_result):
        """_summary_

        Args:
            input_name (array): input name of data
            label_result (array): clustering result 

        Returns:
            dict_result(dict): dictionary type result -> key: nput_name, value: label
        """
        dict_result = dict(zip(input_name, label_result))
        return dict_result
    

    def plot_ts_by_label(self):
        """ plot timeseries style result with representative clustering value.
        
        """
        pass
    

    def plot_label_histogram(self):
        """ plot histogram result with clustered result

        """
        pass