class errorToNaN():
    """
    This class gets data with more NaN. This function converts data identified as errors to NaN. 
    This class finds fake data generated due to network errors, etc., and converts it to NaN.
    
    **Two Outlier Detection Modules**::

            datawithMoreCertainNaN, datawithMoreUnCertainNaN
        
        ``datawithMoreCertainNaN``: Clear Error to NaN

        ``datawithMoreUnCertainNaN``: UnClear Error to NaN
    """
    def __init__(self):
        # Uncertain Remove 에 대한 조절 파라미터 필요 # input parameter로 받아야 함
        # 지금은 강제 True 설정 더 정교해야 Uncertain에 대해서 잘 control 가능해 보임

        # dataRangeInfoManager 대신에 limit_min_max 값을  outlier_param의 값으로 받아들이도록 수정해야 함.
        pass


    def getDataWithCertainNaN(self, data, certain_param):
        """
        This Function converts clear error to NaN.

        Args:
            data (DataFrame): data
            certain_param (Dictionary): certain parameter

        Returns:
            DataFrame: data with More Certain NaN

        Example:
            >>> from Clust.clust.preprocessing.errorDetection import errorToNaN

            >>> min_max = {'max_num': {'in_temp': 80, 'in_humi': 100}, 'min_num': {'in_temp': -40, 'in_humi': 0}}
            >>> abnormal_value_list ={'all':[99.9, 199.9, 299.9, 9999, -99.9, -199.9, -299.9, -9999, -9999.0] }
            >>> certain_param= {'flag': True, 'abnormal_value_list': abnormal_value_list, 'data_min_max_limit': min_max}
            
            >>> data_with_more_certain_nan = errorToNaN.errorToNaN().getDataWithCertainNaN(data, certain_param)

        """
        if 'data_min_max_limit' in certain_param:
            self.limit_min_max = certain_param['data_min_max_limit']   
        else:
            self.limit_min_max  = self.get_default_limit_min_max('air')

        if certain_param['flag'] ==True:  
            from Clust.clust.preprocessing.errorDetection import certainError
            if 'abnormal_value_list' in list(certain_param.keys()):
                abnormal_value_list = certain_param['abnormal_value_list']
            else:
                abnormal_value_list = {'all': [99.9, 199.9, 299.9, 9999, -99.9, -199.9, -299.9, -9999, -9999.0] }
                
            #anomal_value_list=[]
            datawithMoreCertainNaN = certainError.CertainErrorRemove(data, self.limit_min_max, abnormal_value_list).getDataWitoutcertainError()  
        else:
            datawithMoreCertainNaN = data.copy()
        return datawithMoreCertainNaN
    
    def get_default_limit_min_max(self, data_type):
        """
        get data minimum,maximum limit parameter
        
        Args:
            data_type (string) : data type

        Returns:
            Dictionary : data min, max limit information
        """
        from Clust.clust.preprocessing.errorDetection import dataRangeInfo_manager
        limit_min_max = dataRangeInfo_manager.MinMaxLimitValueSet().get_data_min_max_limitSet(data_type)
        return limit_min_max

    def getDataWithUncertainNaN(self, data, uncertain_param):
        """
        This Function converts unclear error to NaN.

        Args:
            data (DataFrame): data
            uncertain_param (Dictionary): uncertain parameter

        Returns:
            DataFrame: data with More UnCertain NaN

        Example:
            >>> from Clust.clust.preprocessing.errorDetection import errorToNaN

            >>> alg_parameter = {'IF_estimators': 100, 'IF_max_samples': 'auto', 'IF_contamination': 0.01, 'IF_max_features': 1.0, 'IF_bootstrap': True}
            >>> uncertain_param = {'flag': True, 
            ...                    'param': {'outlierDetectorConfig': [{'algorithm': 'IF', 'percentile': 99, 'alg_parameter': alg_parameter}]}}

            >>> data_with_more_uncertain_nan = errorToNaN.errorToNaN().getDataWithUncertainNaN(data, uncertain_param)
            
        """    
        if uncertain_param['flag'] == True:
            from Clust.clust.preprocessing.errorDetection import unCertainError
            param = uncertain_param['param']
            data_outlier = unCertainError.unCertainErrorRemove(data, param)
            outlierIndex = data_outlier.getNoiseIndex()
            datawithMoreUnCertainNaN = data_outlier.getDataWithoutUncertainError(outlierIndex)

        else:
            datawithMoreUnCertainNaN = data.copy()
        return datawithMoreUnCertainNaN

