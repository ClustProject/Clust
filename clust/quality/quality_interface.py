import sys
sys.path.append("../")
sys.path.append("../..")
sys.path.append("../../..")

def get_data_result(quality_type, data_input, quality_param):
    """
        Interface function to check data quality.
        * quality_type : data_with_clean_feature
            - Clean Data by each column
                - Delete bad quality column
                - Impute missing data in surviving columns of baseline quality by the NaNInfoCleanData parameter (using linear replacement)
            - input data must be processed and refined by preprocessing(after refining and making more NaN )

        Args:
            quality_type (string) : quality check type
            data_input (dataFrame):  input Data to be handled
            quality_param (dictionary): quality parameter by quality_type

        Returns:
            DataFrame: Clean Data

        Example:
            >>> quality_param = {
            ...    "quality_method":"data_with_clean_feature", 
            ...    "quality_param":{"nan_processing_param":{'type':'num', 'ConsecutiveNanLimit':100, 'totalNaNLimit':1000}}
            ...    }
    """

    nan_processing_param = quality_param['nan_processing_param'] 
    if quality_type =='data_with_clean_feature':
        from Clust.clust.quality.NaN import clean_feature_data
        CMS = clean_feature_data.CleanFeatureData()
        result = CMS.get_cleanData_by_removing_column(data_input, nan_processing_param) 
         
    return result

