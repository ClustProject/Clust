import os
import sys
sys.path.append("../")
sys.path.append("../../")
from multiprocessing import freeze_support

class Trainer():
    def __init__(self, param=None):
        """
        param interpretation
        """
        pass

    def set_param(self, param):
        """ 
        param interpretation

        Args:
            
        """  
        pass

    def set_data(self):
        """
        Data interpretation
        """
        pass

    def get_model(self, model_name):
        """ 
        Get Model

        """  
        pass
    
    def train(self):
        """
        training model Each method should define

        Args:
            data (series): input data for training
        """

        pass
