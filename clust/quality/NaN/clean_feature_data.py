import sys
sys.path.append("../")
sys.path.append("../..")
import pandas as pd
from Clust.clust.quality.NaN import data_remove_byNaN
from Clust.clust.preprocessing import dataPreprocessing

# 특정 datasetd에 대해 품질을 점검하고 각 피쳐별로 이상 수치를 넘는 피쳐 데이터는 제거하고 깨끗한 데이터를 전달
# - multiple dataFrame:getMultipleCleanDataSetsByFeature
# - one dataFrame: getOneCleanDataSetByFeature

class CleanFeatureData:
    def __init__(self, feature_list, frequency):
        """
        """
        self.feature_list = feature_list
        self.frequency = frequency

        self.refine_param = {
            "removeDuplication":{"flag":True},
            "staticFrequency":{"flag":True, "frequency":frequency}
        }
        self.certainParam= {"flag":True}

        self.imputation_param = {
                "flag":True,
                "imputation_method":[{"min":0,"max":10000,"method":"linear" , "parameter":{}}],
                "totalNonNanRatio":5
        }

    def getMultipleCleanDataSetsByDF(self, dataSet, NanInfoForCleanData) :
        """
        This funtion can work by only num type of NaNInfoForCleanData
        :param dataSet: input Data to be handled
        :type dataSet: dictionary
        :param NanInfoForCleanData: selection condition
        :type NanInfoForCleanData: dictionary
        :param duration:  duration, if set duration, make data with full duration, default=None
        :type duration: dictionary

        example:
        >>> NanInfoForCleanData = {'type':'num', 'ConsecutiveNanLimit':1, 'totalNaNLimit':10}

        :returns: self.refinedDataSet, self.FilteredImputedDataSet
        :rtype: 
        """

        self.refinedDataSet={}
        self.FilteredImputedDataSet = {}

        ms_list = dataSet.keys()
        for ms_name in ms_list:
            data = dataSet[ms_name]
            refinedData, DataWithMoreNaN = self._getPreprocessedData(data)
            self.refinedDataSet[ms_name] = refinedData
            
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
                     
        return self.refinedDataSet, self.FilteredImputedDataSet


    def getMultipleCleanDataSetsByFeature(self, dataSet, NanInfoForCleanData, duration=None) :
        """
        refinedDataSet, refinedDataSetName: 간단한 cleaning 진행한 데이터셋
        NaNRemovedDataSet : 품질이 좋지 않은 NaN 값을 다수 포함한 컬럼을 제거한 데이터
        ImputedDataSet: datasetNanRemove의 nan을 임의대로 interpolation한 데이터
        ImputedDataSetName: datasetNanRemove 에 대한 ms 이름


        :param dataSet: input Data to be handled
        :type dataSet: dictionary
        :param NanInfoForCleanData: selection condition
        :type NanInfoForCleanData: dictionary
        :param duration:  duration, if set duration, make data with full duration, default=None
        :type duration: dictionary

        :returns: self.refinedDataSet, self.refinedDataSetName, self.NaNRemovedDataSet, self.ImputedDataSetName, self.ImputedDataSet
        :rtype: 

        prameter ``duration`` example
            docstring::
                {
                    'start_time' : "2021-02-01 00:00:00",
                    'end_time' : "2021-02-04 00:00:00",
                }
        """

        self.refinedDataSet={}
        self.refinedDataSetName={}
        self.NaNRemovedDataSet={}
        self.ImputedDataSet = {}
        self.ImputedDataSetName={}
        for feature in self.feature_list:
            self.refinedDataSet[feature]=[]
            self.refinedDataSetName[feature]=[]
            self.NaNRemovedDataSet[feature]=[]
            self.ImputedDataSet[feature] = []
            self.ImputedDataSetName[feature]=[]

        ms_list = dataSet.keys()
        for ms_name in ms_list:
            #print("=======",ms_name,"=======")
            data = dataSet[ms_name]
            refinedData, NaNRemovedData, ImputedData, finalFlag  = self.getOneCleanDataSetByFeature(data, NanInfoForCleanData, duration)
            for feature in self.feature_list:
                if feature in data.columns:
                    if finalFlag[feature]==-1:
                        pass
                    else: ## final_flag = 0 , 1
                        self.refinedDataSet[feature].append(refinedData[[feature]])
                        self.refinedDataSetName[feature].append(ms_name)
                        if finalFlag[feature] == 1:
                            self.NaNRemovedDataSet[feature].append(NaNRemovedData[[feature]])
                            self.ImputedDataSet[feature].append(ImputedData[[feature]])
                            self.ImputedDataSetName[feature].append(ms_name)
                    
        return self.refinedDataSet, self.refinedDataSetName, self.NaNRemovedDataSet, self.ImputedDataSetName, self.ImputedDataSet


    def getOneCleanDataSetByFeature(self, data, NanInfoForCleanData, duration=None) :
        """
        This function gets CleanDataSet by Feature

        :param data: input Data to be handled
        :type data: dataFrame
        :param NanInfoForCleanData:  selection condition
        :type NanInfoForCleanData: dictionary
        :param duration:  duration, if set duration, make data with full duration, default=None
        :type duration: dictionary

        :returns: refinedData
        :rtype: dataframe
        :returns: NaNRemovedData
        :rtype: dataFrame
        :returns: ImputedData
        :rtype: dataFrame
        :returns: finalFlag
        :rtype: dictionary (-1: no data, 0:useless data, 1:useful data)

        prameter ``duration`` example
        docstring::
            {
                'start_time' : "2021-02-01 00:00:00",
                'end_time' : "2021-02-04 00:00:00",
            }
        """

        if duration:
            data = self._setDataWithSameDuration(data, duration)
        refinedData, DataWithMoreNaN = self._getPreprocessedData(data)

        finalFlag = {}
        NaNRemovedData = {}
        ImputedData = {}
        DRN = data_remove_byNaN.DataRemoveByNaNStatus()
        NaNRemovedData = DRN.removeNaNData(DataWithMoreNaN, NanInfoForCleanData)
        ImputedData = NaNRemovedData.copy()
              
        for feature in self.feature_list:
            finalFlag[feature] = -1
            if (feature in data.columns) and (len(data[feature]) >0) and ( data[feature].isna().sum() != len(data[feature])): # refined_data 가 존재하고, 기존 data에 feature(column)이 속해 있을 때
                finalFlag[feature] = 0
                if feature in NaNRemovedData.columns: # NaN의 limit을 넘은 컬럼 삭제 후, 컬럼이 남아있으면
                    NaNRemovedData_feature = NaNRemovedData[[feature]]
                    finalFlag[feature] = 1
                    MDP = dataPreprocessing.DataPreprocessing()
                    ImputedData[feature] = MDP.get_imputedData(NaNRemovedData_feature, self.imputation_param)
            else:
                finalFlag[feature] = -1
        
        return refinedData, NaNRemovedData, ImputedData, finalFlag

    
    def _getPreprocessedData(self, data):
        """
        This function produced cleaner data with parameter

        :param data: input Data to be handled
        :type data: dataFrame with only one feature

        :returns: refinedData
        :rtype: dataFrame

        :returns: dataWithMoreNaN
        :rtype: dataFrame
        """
        refined_data = pd.DataFrame()
        datawithMoreCertainNaN = pd.DataFrame()
        
        if len(data)>0:
            #1. Preprocessing (Data Refining/Static Frequency/OutlierDetection)
            MDP = dataPreprocessing.DataPreprocessing()
            refined_data = MDP.get_refinedData(data, self.refine_param)
            from Clust.clust.preprocessing.errorDetection.errorToNaN import errorToNaN 
            datawithMoreCertainNaN = errorToNaN().getDataWithCertainNaN(refined_data, self.certainParam)
        
        return refined_data, datawithMoreCertainNaN

                
    # self.query_start_time, self.query_end_time 문제 이슈
    def _setDataWithSameDuration(self, data, duration):
        # Make Data with Full Duration [query_start_time ~ query_end_time]
        
        start_time =duration['start_time']
        end_time = duration['end_time']
        # print("setDataWithSameDuration", start_time, end_time)
        if len(data)>0:
            #2. Make Full Data(query Start ~ end) with NaN
            data.index = data.index.tz_localize(None) # 지역정보 없이 시간대만 가져오기
            new_idx = pd.date_range(start = start_time, end = (end_time- self.frequency), freq = self.frequency)
            new_data = pd.DataFrame(index= new_idx)
            new_data.index.name ='time' 
            data = new_data.join(data) # new_data에 data를 덮어쓴다
        return data

        