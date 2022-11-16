import numpy as np

class PeriodicFeature():
    """
    This class produces extended feature by periodically feature from data
    """
    def __init__(self):
        pass
    
    
    def extendCosSinFeature(self, origin_df, col_name, period, start_num=0, dropOriginal=False):
        """
        This function generate transformed feature by period. Each column must have a specific period.
        It calculats sine and cosine transform value of the given feature.
        Example
        -------
        >>> from clust.transformation.featureExtension.periodicFeatureExtension import PeriodicFeature
        >>> PF = PeriodicFeature()
        >>> df_extended =  PF.extendCosSinFeature(data, 'hour', 24, 0, True)

        original_df: pandas.DataFrame
            original Input DataFrame
        columnName: str
            Column name to convert
        period: int
            the period of column
        dropOriginal : bool, default = False
            When set to True, it drops original column features 
        Returns:
            extended_df (pandas.DataFrame), New dataFrmae with one-hot-encoded features
        """ 
        extended_df = origin_df.copy()
        
        kwargs = {
            f'sin_{col_name}' : lambda x: np.sin(2*np.pi*(extended_df[col_name]-start_num)/period),
            f'cos_{col_name}' : lambda x: np.cos(2*np.pi*(extended_df[col_name]-start_num)/period)    
                }
        extended_df = extended_df.assign(**kwargs)
        if dropOriginal==True:
            extended_df = extended_df.drop(columns=[col_name]) 
            
        return extended_df
    