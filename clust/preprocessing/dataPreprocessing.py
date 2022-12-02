import sys
import os
sys.path.append("../")
sys.path.append("../..")

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
        This function gets refined data with static frequency, without redundency data. 
        It refines data adaptively depending on flag status. (removeDuplication, staticFrequency)
        ``removeDuplication`` :It removes duplicated data.
        ``staticFrequency`` :The data will have a constant timestamp index. 


        Args:
            data (DataFrame): data
            refine_param (Dictionary): refinement parameter
            
        Returns:
            DataFrame: refinedData, refined DataFrame output


        refine_param additional info: 
            >>> refine_param['removeDuplication']={'flag':(Boolean)} 
            >>> refine_param['staticFrequency'] ={'flag':(Boolean), 'frequency':[None|timeinfo]}
            >>> refine_param['ststicFreeuncy']['frequnecy'] == None -> infer original frequency and make static time stamp.


        Example:

            >>> from clust.preprocessing.dataPreprocessing import DataPreprocessing
            >>> refine_param = {'removeDuplication': {'flag': True}, 'staticFrequency': {'flag': True, 'frequency': None}}
            >>> refine_param2 = {'removeDuplication': {'flag': True}, 'staticFrequency': {'flag': True, 'frequency': "3H"}}
            >>> refinementData = DataPreprocessing().get_refinedData(data, refine_param)

        """
        result = data.copy()
        if refine_param['removeDuplication']['flag']== True:
            from Clust.clust.preprocessing.refinement import redundancy
            result = redundancy.ExcludeRedundancy().get_result(result)

        if refine_param['staticFrequency']['flag'] == True:
            from Clust.clust.preprocessing.refinement import frequency
            inferred_freq = refine_param['staticFrequency']['frequency']
            result = frequency.RefineFrequency().get_RefinedData(result, inferred_freq)

        self.refinedData = result
        return self.refinedData
    
    def get_errorToNaNData(self, data, outlier_param):

        """
        This function gets data with more NaN. This function converts data identified as errors to NaN. 
        This module finds fake data generated due to network errors, etc., and converts it to NaN.

        Args:
            data (DataFrame): data
            outlier_param (Dictionary): outlier handling parameter
            
        Returns:
            DataFrame: datawithMoreCertainNaN, datawithMoreUnCertainNaN

        Example:

            >>> uncertainErrorParam = {
                # TODO define }
            >>> outlier_param = {'certainErrorToNaN': {'flag': True}, 'unCertainErrorToNaN': uncertainErrorParam}
            >>> datawithMoreCertainNaN, datawithMoreUnCertainNaN = DataPreprocessing().get_errorToNaNData(data, outlier_param)


        **Two Outlier Detection Modules**::

            datawithMoreCertainNaN, datawithMoreUnCertainNaN
        
        ``datawithMoreCertainNaN``: Clear Error to NaN

        ``datawithMoreUnCertainNaN``: UnClear Error to NaN

        
            
        """
        from Clust.clust.preprocessing.errorDetection import errorToNaN
        self.datawithMoreCertainNaN = errorToNaN.errorToNaN().getDataWithCertainNaN(data, outlier_param['certainErrorToNaN'])
        self.datawithMoreUnCertainNaN = errorToNaN.errorToNaN().getDataWithUncertainNaN(self.datawithMoreCertainNaN, outlier_param['unCertainErrorToNaN'])
        return self.datawithMoreCertainNaN, self.datawithMoreUnCertainNaN

    def get_imputedData(self, data, imputation_param):
        """ Get imputed data

        Args:
            data (DataFrame): input data
            refine_param (json): imputation_param
            
        Returns:
            DataFrame: New Dataframe after imputation
        
        Example:

            >>> imputation_param = {'flag': True, 
            ...                     'imputation_method': [{'min': 0, 'max': 3, 'method': 'KNN', 'parameter': {}}, '
            ...                                           {'min': 4, 'max': 6, 'method': 'mean', 'parameter': {}}], 
            ...                     'totalNonNanRatio': 80}
            >>> output = DataPreprocessing().get_imputedData(data, outlier_param)

        """
        self.imputedData = data.copy()
        if imputation_param['flag'] == True:
            from Clust.clust.preprocessing.imputation import Imputation
            self.imputedData = Imputation.SerialImputation().get_dataWithSerialImputationMethods(self.imputedData, imputation_param)

        return self.imputedData

    # Add New Function

class DataProcessing(DataPreprocessing):
    '''This class provides funtion having packged preprocessing procedure.
    '''
    def __init__(self, process_param):
        '''Set process_param related to each preprocessing module.

        Args:
            process_param (json): process_param

        Example:

            >>> process_param={
            ...                 "refine_param": {},
            ...                 "outlier_param": {},
            ...                 "imputation_param":{}
            ...                 }

        '''
        self.refine_param = process_param['refine_param']
        self.outlier_param = process_param['outlier_param']
        self.imputation_param = process_param['imputation_param']
    
    def preprocessing(self, input_data, flag):
        """ Produces only one clean data with one preprocessing module.

        Args:
            data (DataFrame): input data
            flag (string): preprocessing name
            
        Returns:
            DataFrame: New Dataframe after one preprocessing (flag)
        
        Example:

            >>> output = DataProcessing().preprocessing(data, 'refine')
            
        """
        if flag == 'refine':
            result = self.get_refinedData(input_data, self.refine_param)
        elif flag =='errorToNaN':
            result = self.get_errorToNaNData(input_data, self.outlier_param)
        elif flag == 'imputation':
            result = self.get_imputedData(input_data, self.imputation_param)
        elif flag == 'all':
            result = self.all_preprocessing(input_data)
        return result

    def all_preprocessing(self, input_data):
        """ Produces partial Processing data depending on process_param

        Args:
            input_data (DataFrame): input data
            
        Returns:
            json: New Dataframe after preprocessing according to the process_param
        
        Example:

            >>> output = DataProcessing(process_param).all_preprocessing(data)
            
        """
        ###########
        refined_data = self.get_refinedData(input_data, self.refine_param)
        ###########
        datawithMoreCertainNaN, datawithMoreUnCertainNaN = self.get_errorToNaNData(refined_data, self.outlier_param)
        ###########
        imputed_data = self.get_imputedData(datawithMoreUnCertainNaN, self.imputation_param)
        ###########
        result ={'original':input_data, 'refined_data':refined_data, 'datawithMoreCertainNaN':datawithMoreCertainNaN,
        'datawithMoreUnCertainNaN':datawithMoreUnCertainNaN, 'imputed_data':imputed_data}
        return result

    ## Get Multiple output
    def multiDataset_all_preprocessing(self, multiple_dataset):
        """ Produces multiple DataFrame Processing result depending on process_param

        Args:
            input_data (json): multiple_dataset - (having DataFrame value) 
            
        Returns:
            json: json having New Dataframe after preprocessing according to the process_param
        
        Example:

            >>> output = DataProcessing(process_param).multiDataset_all_preprocessing(multiple_dataset)
        """
        output={}
        for key in list(multiple_dataset.keys()):
            output[key] = self.all_preprocessing(multiple_dataset[key])
        return output

