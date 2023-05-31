import sys
sys.path.append("../")
sys.path.append("../..")
import pandas as pd
from Clust.clust.quality.NaN import data_remove_byNaN
from Clust.clust.preprocessing import dataPreprocessing

# 특정 datasetd에 대해 품질을 점검하고 각 피쳐별로 이상 수치를 넘는 피쳐 데이터는 제거하고 깨끗한 데이터를 전달
# - multiple dataFrame:get_multiple_clean_datasets_by_feature
# - one dataFrame: get_one_clean_dataset_by_feature

""" get_one_clean_dataset_by_feature, getMultipleCleanDataSetsByDF 안쓰는듯."""

class CleanFeatureData:
    def __init__(self):
        """
        - Clean Data by each column
        - Delete bad quality column
        - recover column with moderate quality by NaNInfoCleanData partameter (use linear imputation)
        
        """
        self.imputation_param = {
                "flag":True,
                "imputation_method":[{"min":0,"max":10000,"method":"linear" , "parameter":{}}],
                "total_non_NaN_ratio":5
        }
    """
    def getMultipleCleanDataSetsByDF(self, dataSet, NanInfoForCleanData) :
        
        This funtion can work by only num type of NaNInfoForCleanData

        Args:
            dataSet (dictionary):  input Data to be handled
            NanInfoForCleanData (dictionary): selection condition 
            duration (dictionary): duration
            
        Note
        ---------
        if set duration, make data with full duration, default=None


        Example:

            >>> NanInfoForCleanData = {'type':'num', 'ConsecutiveNanLimit':1, 'totalNaNLimit':10}

        Returns:
            Dict: self.refinedDataSet, self.FilteredImputedDataSet

        

        self.FilteredImputedDataSet = {}

        ms_list = dataSet.keys()
        for ms_name in ms_list:
            data = dataSet[ms_name]
            
            if len(data)>0:  
                totalNanRowCount = DataWithMoreNaN.isnull().any(axis=1).sum()
                if totalNanRowCount < NanInfoForCleanData['totalNaNLimit']:
                    consecutiveNanCountMap = DataWithMoreNaN.isnull().any(axis=1).astype(int).groupby(DataWithMoreNaN.notnull().any(axis=1).astype(int).cumsum()).cumsum()
                    ConsecutiveNanLimitNum = NanInfoForCleanData['ConsecutiveNanLimit']
                    if (consecutiveNanCountMap > ConsecutiveNanLimitNum).any():
                        
                        pass
                    else:
                        imputedData = dataPreprocessing.DataPreprocessing().get_imputedData(DataWithMoreNaN, self.imputation_param)
                        self.FilteredImputedDataSet[ms_name] = imputedData     
                     
        return self.FilteredImputedDataSet

    """
    def get_multiple_clean_datasets_by_feature(self, dataSet, NanInfoForCleanData, duration=None) :
        """
        - refinedDataSet, refinedDataSetName: 간단한 cleaning 진행한 데이터셋
        - NaNRemovedDataSet : 품질이 좋지 않은 NaN 값을 다수 포함한 컬럼을 제거한 데이터
        - ImputedDataSet: datasetNanRemove의 nan을 임의대로 interpolation한 데이터
        - ImputedDataSetName: datasetNanRemove 에 대한 ms 이름

        Args:
            dataSet (dictionary):  input Data to be handled
            NanInfoForCleanData (dictionary): selection condition 
            duration (dictionary): duration
            
        Returns:
            Dict: self.refinedDataSet, self.refinedDataSetName, self.NaNRemovedDataSet, self.ImputedDataSetName, self.ImputedDataSet

        Example:

            >>> duration = { 'start_time' : "2021-02-01 00:00:00",
            ...               'end_time' : "2021-02-04 00:00:00" }

        """
        self.imputed_data_set = pd.DataFrame()

        for column_name in list(dataSet.columns):
            data = dataSet[[column_name]]
            imputed_data  = self.get_one_clean_dataset_by_feature(data, NanInfoForCleanData, duration)
            self.imputed_data_set = pd.concat([self.imputed_data_set, imputed_data], axis=1)
                    
        return self.imputed_data_set


    def get_one_clean_dataset_by_feature(self, data, NanInfoForCleanData, duration=None) :
        """
        This function gets CleanDataSet by Feature

        Args:
            dataSet (dictionary):  input Data to be handled
            NanInfoForCleanData (dictionary): selection condition 
            duration (dictionary): duration

        Returns:
            dataframe: refinedData, NaNRemovedData, ImputedData

        Returns:
            dictionary: finalFlag (-1: no data, 0:useless data, 1:useful data)


        Example:

            >>> duration = { 'start_time' : "2021-02-01 00:00:00",
            ...               'end_time' : "2021-02-04 00:00:00" }

        """
        nan_removed_data = self.get_cleanData_by_removing_column(data, NanInfoForCleanData)
        MDP = dataPreprocessing.DataPreprocessing()
        imputed_data= MDP.get_imputedData(nan_removed_data, self.imputation_param)
        
        return imputed_data
    
    def get_cleanData_by_removing_column(self, data, NanInfoForCleanData) :
        """
        - Clean Data by each column
            - Delete bad quality column
            - Impute missing data in surviving columns of baseline quality by the NaNInfoCleanData parameter (using linear replacement)
        - input data must be processed and refined by preprocessing(after refining and making more NaN )

        Args:
            data (np.DataFrame):  input Data to be handled
            NanInfoForCleanData (dictionary): selection condition 

        Returns:
            DataFrame: filtered and imputed data

        """

        DRN = data_remove_byNaN.DataRemoveByNaNStatus()
        nan_removed_data = DRN.removeNaNData(data, NanInfoForCleanData)

        print(len(data.columns), "--->", len(nan_removed_data.columns))

        return nan_removed_data


