class Inference():
    def __init__(self):
        pass
        
    def set_param(self, param):
        """
        Set Parameter for Test & Inference

        Args:
        param(dict): train parameter


        Example:

            >>> param = { 'num_layers': 2, 
            ...            'hidden_size': 64, 
            ...            'dropout': 0.1,
            ...            'bidirectional': True,
            ...            "lr":0.0001,
            ...            "device":"cpu",
            ...            "batch_size":16,
            ...            "n_epochs":10    }
            

        """
        pass


    def set_data(self, data, window_num):
        """
        set data for test or inference & transform data

        Args:
            data (dataframe): Test or Inference data
            window_num (integer) : window size

        Example:

            >>> if 1) Test
            ...     set_data(test_X, test_y, window_num)
            ...         test_X : test X data
            ...         test_y : test y data
            ...         window_num : window size

            >>> if 2) Inference
            ...     set_data(test_X, window_num)
            ...         test_X : inference data
            ...         window_num : window size

        """
        pass


    def get_result(self, model):
        """
        Predict RegresiionResult based on model result

        Args:
            model (model) : load train model

        Returns:
            preds (ndarray) : Test or Inference result data
        
        """
        pass



#----------------------------------------------------------------------------------------
    def set_param(self, param):
        """
        Set Parameter for Test & Inference

        Args:
        param(dict): train parameter


        Example:

            >>> param = { 'num_layers': 2, 
            ...            'hidden_size': 64, 
            ...            'dropout': 0.1,
            ...            'bidirectional': True,
            ...            "lr":0.0001,
            ...            "device":"cpu",
            ...            "batch_size":16,
            ...            "n_epochs":10    }
            

        """
        pass


    def set_test_data(self, data, window_num):
        """
        set data for test or inference & transform data

        Args:
            data (dataframe): Test or Inference data
            window_num (integer) : window size

        Example:

            >>>         set_data(test_X, test_y, window_num)
            ...         test_X : test X data
            ...         test_y : test y data
            ...         window_num : window size

        """
        pass


    def get_test_result(self, model):
        """
        Predict RegresiionResult based on model result

        Args:
            model (model) : load train model

        Returns:
            preds (ndarray) : prediction data
            trues (ndarray) : original data
        
        """
        pass



    def set_inference_data(self, data, window_num):
        """
        set data for test or inference & transform data

        Args:
            data (dataframe): Test or Inference data
            window_num (integer) : window size

        Example:

            >>>         set_data(test_X, window_num)
            ...         test_X : inference data
            ...         window_num : window size

        """
        pass


    def get_inference_result(self, model):
        """
        Predict RegresiionResult based on model result

        Args:
            model (model) : load train model

        Returns:
            preds (ndarray) : Inference result data
        
        """
        pass
