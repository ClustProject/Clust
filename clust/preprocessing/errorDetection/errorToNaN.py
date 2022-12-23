class errorToNaN():
    def __init__(self):
        # Uncertain Remove 에 대한 조절 파라미터 필요 # input parameter로 받아야 함
        # 지금은 강제 True 설정 더 정교해야 Uncertain에 대해서 잘 control 가능해 보임

        # dataRangeInfoManager 대신에 limit_min_max 값을  outlier_param의 값으로 받아들이도록 수정해야 함.
        pass

    def getDataWithCertainNaN(self, data, CertainParam):  
        
        self.limit_min_max = CertainParam['data_min_max_limit']   

        if CertainParam['flag'] ==True:  
            from Clust.clust.preprocessing.errorDetection import certainError
            anomal_value_list=[99.9, 199.9, 299.9, 9999, -99.9, -199.9, -299.9, -9999, -9999.0] 
            #anomal_value_list=[]
            datawithMoreCertainNaN = certainError.CertainErrorRemove(data, self.limit_min_max, anomal_value_list).getDataWitoutcertainError()  
        else:
            datawithMoreCertainNaN = data.copy()
        return datawithMoreCertainNaN
    
    def getDataWithUncertainNaN(self, data, uncertainParam):    
        if uncertainParam['flag'] == True:
            from Clust.clust.preprocessing.errorDetection import unCertainError
            param = uncertainParam['param']
            data_outlier = unCertainError.unCertainErrorRemove(data, param)
            outlierIndex = data_outlier.getNoiseIndex()
            datawithMoreUnCertainNaN = data_outlier.getDataWithoutUncertainError(outlierIndex)

        else:
            datawithMoreUnCertainNaN = data.copy()
        return datawithMoreUnCertainNaN




"""
class errorToNaN():
    def __init__(self, data_type='air'):
        # Uncertain Remove 에 대한 조절 파라미터 필요 # input parameter로 받아야 함
        # 지금은 강제 True 설정 더 정교해야 Uncertain에 대해서 잘 control 가능해 보임

        # dataRangeInfoManager 대신에 limit_min_max 값을  outlier_param의 값으로 받아들이도록 수정해야 함.
        self.limit_min_max = self.dataRangeInfoManager(data_type)

    def dataRangeInfoManager(self, data_type):
        from Clust.clust.preprocessing.errorDetection import dataRangeInfo_manager
        limit_min_max = dataRangeInfo_manager.MinMaxLimitValueSet().get_data_min_max_limitSet(data_type)
        return limit_min_max


    def getDataWithCertainNaN(self, data, CertainParam):
        if CertainParam['flag'] ==True:  
            from Clust.clust.preprocessing.errorDetection import certainError
            anomal_value_list=[99.9, 199.9, 299.9, 9999, -99.9, -199.9, -299.9, -9999, -9999.0] 
            #anomal_value_list=[]
            datawithMoreCertainNaN = certainError.CertainErrorRemove(data, self.limit_min_max, anomal_value_list).getDataWitoutcertainError()  
        else:
            datawithMoreCertainNaN = data.copy()
        return datawithMoreCertainNaN
    
    def getDataWithUncertainNaN(self, data, uncertainParam):    
        if uncertainParam['flag'] == True:
            from Clust.clust.preprocessing.errorDetection import unCertainError
            param = uncertainParam['param']
            data_outlier = unCertainError.unCertainErrorRemove(data, param)
            outlierIndex = data_outlier.getNoiseIndex()
            datawithMoreUnCertainNaN = data_outlier.getDataWithoutUncertainError(outlierIndex)

        else:
            datawithMoreUnCertainNaN = data.copy()
        return datawithMoreUnCertainNaN

"""
