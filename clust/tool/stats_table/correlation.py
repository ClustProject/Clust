class Correlation():
    """
    # Description
        get Correlation Matrix and related information

    """
    def __init__(self, data):
        """
        # Description
            set input data

        # Args
            - data (_pd.dataFrame_)

        """
        self.data = data

    def getCorrelationMatrix(self):
        """
        # Description
            get pearson correlation Matrix

        # Returns
            - corrMtx (_pd.DataFrame_) : correlation Matrix

        """
        self.corrMtx = self.data.corr()

        return self.corrMtx

    def _get_redundant_pairs(self):        
        """
        # Description
            Get diagonal and lower triangular pairs of correlation matrix

        # Returns
            - pairs_to_drop (_pd.DataFrame_) : pairs of correlation matrix
        """
        pairs_to_drop = set()
        cols = self.data.columns

        for i in range(0, self.data.shape[1]):
            for j in range(0, i+1):
                pairs_to_drop.add((cols[i], cols[j]))

        return pairs_to_drop

    def get_top_abs_correlations(self, ranking=5, target = None):
        """
        # Description
            Get the highest correlation pairs

        # Args
            - data (_pd.dataFrame_) : data
            - ranking (_Integer_) : variable to specify the top Nth value
            - target (_String_) : If target is specified, only results related to target name are provided

        # Returns
            - result (_series of DataFrame_) : pairs of correlation matrix

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
    