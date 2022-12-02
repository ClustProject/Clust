class Correlation():
    """
    get Correlation Matrix and related information
    """
    def __init__(self, data):
        """
        set input data

        Args:
            data (dataFrame): data
        """
        self.data = data

    def getCorrelationMatrix(self):
        """
        get pearson correlation Matrix

        Returns:
            DataFrame: correlation Matrix
        """
        self.corrMtx = self.data.corr()
        return self.corrMtx

    def _get_redundant_pairs(self):
        """
        Get diagonal and lower triangular pairs of correlation matrix

        Returns:
            DataFrame: pairs of correlation matrix
        """
        pairs_to_drop = set()
        cols = self.data.columns
        for i in range(0, self.data.shape[1]):
            for j in range(0, i+1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop

    def get_top_abs_correlations(self, ranking=5, target = None):
        """
        Get the highest correlation pairs

        Args:
            data (dataFrame): data
            ranking (Integer): variable to specify the top Nth value
            target (String): If target is specified, only results related to target name are provided

        Returns:
            series of DataFrame: pairs of correlation matrix

        """
        au_corr = self.getCorrelationMatrix().abs().unstack()
        labels_to_drop = self._get_redundant_pairs()
        au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
        if target:
            result = au_corr.iloc[(au_corr.index.get_level_values(0) == target) | (au_corr.index.get_level_values(1) == target)]
            result = result[0:ranking]
        else:
            result = au_corr[0:ranking]
        return result
    