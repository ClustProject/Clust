class ExcludeRedundancy():
    """ Exclude Redundancy data
    """
    def __init__(self):
        pass
    
    def get_result(self, data):
        """ Get Clean Data without redundency using all data preprocessing functions.

        :param data: input data
        :type data: DataFrame 

        :return: output data
        :rtype: DataFrame

        example
            >>> output = ExcludeRedundancy().get_result(data)
        """
        self.result = self.RemoveDuplicateData(data)
        return self.result

    def RemoveDuplicateData(self, data):
        """ Return clean data removing duplicate row and/or column

        :param data: input data
        :type data: DataFrame 

        :return: output data
        :rtype: DataFrame

        example
            >>> output = ExcludeRedundancy().RemoveDuplicateData(data)
        """
        ## 
        # duplicated column remove
        data = data.loc[:, ~data.columns.duplicated()]
        
        # duplicated Index Drop
        data = data.sort_index()
        result = data[~data.index.duplicated(keep='first')]
        return result

 