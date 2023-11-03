import sys
import os
sys.path.append("../")
sys.path.append("../..")
import pandas as pd
class DataPreprocessing():
    """
    This class has interfaces of Data Preprocessing.

    **Data Preprocessing Modules**::

            Refine Data, Remove Outlier, Impute Missing Data
    """
    
    
    def __init__(self):
        pass
    
    def get_refinedData(self, data, refine_param):
        """
        - This function gets refined data with static frequency, without redundency data. 
        - It refines data adaptively depending on flag status. (remove_duplication, static_frequency)
        - remove_duplication :It removes duplicated data.
        - static_frequency :The data will have a constant timestamp index. 

        Args:
            data (DataFrame): data
            refine_param (Dictionary): refinement parameter
            
        Returns:
            DataFrame: refinedData, refined DataFrame output

        **refine_param additional info**::

            >>> refine_param["remove_duplication"]={'flag':(Boolean)} 
            >>> refine_param["static_frequency"] ={'flag':(Boolean), 'frequency':[None|timeinfo]}
            >>> refine_param['ststicFreeuncy']['frequnecy'] == None -> infer original frequency and make static time stamp.

        Example:

            >>> from clust.preprocessing.dataPreprocessing import DataPreprocessing
            >>> refine_param = {"remove_duplication": {'flag': True}, "static_frequency": {'flag': True, 'frequency': None}}
            >>> refine_param2 = {"remove_duplication": {'flag': True}, "static_frequency": {'flag': True, 'frequency': "3H"}}
            >>> refinementData = DataPreprocessing().get_refinedData(data, refine_param)
        
        """
        result = data.copy()
        if refine_param["remove_duplication"]['flag']== True:
            from Clust.clust.preprocessing.refinement import redundancy
            result = redundancy.ExcludeRedundancy().get_result(result)

        if refine_param["static_frequency"]['flag'] == True:
            from Clust.clust.preprocessing.refinement import frequency
            inferred_freq = refine_param["static_frequency"]['frequency']
            result = frequency.RefineFrequency().get_RefinedData(result, inferred_freq)

        self.refinedData = result
        return self.refinedData
    
    def get_errorToNaNData(self, data, outlier_param):

        """
        - This function gets data with more NaN. 
        - This function converts data identified as errors to NaN. 
        - This module finds fake data generated due to network errors, etc., and converts it to NaN.

        Args:
            data (DataFrame): data
            outlier_param (Dictionary): outlier handling parameter
            
        Returns:
            DataFrame: datawithMoreCertainNaN, datawithMoreUnCertainNaN

        Example:

            >>> uncertainErrorParam = {'flag': True, 
            ...                        'param': {'outlierDetectorConfig': [{'algorithm': 'IF', 'percentile': 99, 'alg_parameter': algorithm_param}]}}
            >>> outlier_param = {'certain_error_to_NaN': {'flag': True}, 'uncertain_error_to_NaN': uncertainErrorParam}
            >>> datawithMoreCertainNaN, datawithMoreUnCertainNaN = DataPreprocessing().get_errorToNaNData(data, outlier_param)


        **Two Outlier Detection Modules**::

            datawithMoreCertainNaN, datawithMoreUnCertainNaN
            datawithMoreCertainNa: Clear Error to NaN
            datawithMoreUnCertainNaN: UnClear Error to NaN

        
            
        """
        from Clust.clust.preprocessing.errorDetection import errorToNaN
        self.datawithMoreCertainNaN = errorToNaN.errorToNaN().getDataWithCertainNaN(data, outlier_param['certain_error_to_NaN'])
        self.datawithMoreUnCertainNaN = errorToNaN.errorToNaN().getDataWithUncertainNaN(self.datawithMoreCertainNaN, outlier_param['uncertain_error_to_NaN'])
        return self.datawithMoreCertainNaN, self.datawithMoreUnCertainNaN

    def get_imputedData(self, data, imputation_param):
        """ 
        Get imputed data

        Args:
            data (DataFrame): input data
            imputation_param (json): imputation_param
            
        Returns:
            DataFrame: New Dataframe after imputation
        
        Example:

            >>> imputation_param = {'flag': True, 
            ...                     'imputation_method': [{'min': 0, 'max': 3, 'method': 'KNN', 'parameter': {}}, '
            ...                                           {'min': 4, 'max': 6, 'method': 'mean', 'parameter': {}}], 
            ...                     'total_non_NaN_ratio': 80}
            >>> output = DataPreprocessing().get_imputedData(data, outlier_param)

        """
        self.imputedData = data.copy()
        if imputation_param['flag'] == True:
            from Clust.clust.preprocessing.imputation import Imputation
            print(data.shape)
            self.imputedData = Imputation.SerialImputation().get_dataWithSerialImputationMethods(self.imputedData, imputation_param)

        return self.imputedData

    def get_smoothed_data(self, data, smoothing_param):
        """ 
        Get smoothed data

        Args:
            data (DataFrame): input data
            smoothing_param (json): smoothing_param 
            
        Returns:
            DataFrame: New Dataframe after smoothing
        
        Example:

            >>> smoothing_param = {'flag': True, 'emw_param':0.3} #emw parameter. Defaults to 0.3.


        """
        if smoothing_param['flag']==True:
            data = data.ewm(com=smoothing_param['emw_param'] ).mean()
        return data
    
    def get_scaling_data(self, data, scaling_param):
        """ 
        Get scaled data

        Args:
            data (DataFrame): input data
            scaling_param (json): scaling_param with scaling method  (method: minmax, standard, maxabs, robust) 
            
        Returns:
            DataFrame: New Dataframe after scaling
        
        Example:

            >>> scaling_param = {'flag': True, 'method':'minmax'} 


        """
        # TODO all scaler defin
        if scaling_param['flag']==True:
            method = scaling_param['method']
            from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
            scalers = {
                "minmax": MinMaxScaler,
                "standard": StandardScaler,
                "maxabs": MaxAbsScaler,
                "robust": RobustScaler,
            }
            scaler = scalers.get(method.lower())()
            data = pd.DataFrame(scaler.fit_transform(data), columns=list(data.columns), index = data.index)   
        
        return data
        
    # Add New Function
