import sys
sys.path.append("../")
sys.path.append("../../")

class Train():
    def __init__(self, param=None):
        """
        param interpretation
        """
        pass

    def set_param(self, param):
        """ 
        param interpretation for training

        Args:
            param (dictionary) : parameter for train


        Example:

            >>> param = { 'arg1': 2, 
            ...           'arg2': 64, 
            ...           'arg3': 0.1,
            ...            ...
            ...            ...        }
            
        """  
        pass


    def set_data(self, data, window_num=0):
        """
        set train, val data & transform data for training

        Args:
            data (dataframe): train & val data
            window_num (integer): window size


        Example:

            >>> set_data(train_X, train_y, val_X, val_y, window_num)
            ... train_X : train X data
            ... train_y : train y data
            ... val_X : validation X data
            ... val_y : validation y data
            ... window_num : window size

        """
        pass


    def set_model(self, model_method):
        """ 
        Build model and return initialized model for selected model_name

        Args:
            model_method (string): model method name    
        """  
        pass

    def train(self):
        """
        Train model and return model

        Returns:
            model: train model

        """
        pass
