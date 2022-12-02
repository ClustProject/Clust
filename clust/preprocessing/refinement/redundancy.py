class ExcludeRedundancy():
    """ Exclude Redundancy data
    """
    def __init__(self):
        pass
    
    def get_result(self, data):
        """ Get Clean Data without redundency using all data preprocessing functions.

        Args:
            data (DataFrame): input data
            
        Returns:
            DataFrame: result, output data

        Example:

            >>> output = ExcludeRedundancy().get_result(data)
        """

        data = data.loc[:, ~data.columns.duplicated()]
        # duplicated Index Drop
        data = data.sort_index()
        result = data[~data.index.duplicated(keep='first')]
        
        return result

 