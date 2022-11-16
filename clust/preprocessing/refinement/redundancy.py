class ExcludeRedundancy():
    """ Exclude Redundancy data
    """
    def __init__(self):
        pass
    
    def get_result(self, data):
        """ Get Clean Data without redundency using all data preprocessing functions.

        :param data: input data
        :type data: pandas.DataFrame 

        :return: result, output data
        :rtype: pandas.DataFrame

        example
            >>> output = ExcludeRedundancy().get_result(data)
        """

        data = data.loc[:, ~data.columns.duplicated()]
        # duplicated Index Drop
        data = data.sort_index()
        result = data[~data.index.duplicated(keep='first')]
        
        return result

 