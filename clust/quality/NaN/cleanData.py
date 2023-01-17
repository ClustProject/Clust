import sys
sys.path.append("../")
sys.path.append("../..")
import pandas as pd
from Clust.clust.quality.NaN import data_remove_byNaN
from Clust.clust.preprocessing import dataPreprocessing

# 특정 datasetd에 대해 품질을 점검하고 각 피쳐별로 이상 수치를 넘는 피쳐 데이터는 제거하고 깨끗한 데이터를 전달
# - multiple dataFrame:getMultipleCleanDataSetsByFeature
# - one dataFrame: getOneCleanDataSetByFeature


class CleanData:
    def __init__(self):
        pass


    def get_cleanData_by_removing_column(self, data, NanInfoForCleanData, duration=None) :
        """
        - Clean Data by each column
            - Delete bad quality column
            - Impute missing data in surviving columns of baseline quality by the NaNInfoCleanData parameter (using linear replacement)
        - input data must be processed and refined by preprocessing(after refining and making more NaN )

        Args:
            data (np.DataFrame):  input Data to be handled
            NanInfoForCleanData (dictionary): selection condition 

        Returns:
            ImputedData(dataframe): filtered and imputed data

        """
        self.imputation_param = {
                "flag":True,
                "imputation_method":[{"min":0,"max":10000,"method":"linear" , "parameter":{}}],
                "totalNonNanRatio":5
        }
        DRN = data_remove_byNaN.DataRemoveByNaNStatus()
        nan_removed_data = DRN.removeNaNData(data, NanInfoForCleanData)
        MDP = dataPreprocessing.DataPreprocessing()
        imputed_data= MDP.get_imputedData(nan_removed_data, self.imputation_param)

        print(len(data.columns), "--->", len(imputed_data.columns))

        return imputed_data


    """             
    # self.query_start_time, self.query_end_time 문제 이슈
    def _get_multipleDF_sameDuration(self, data, duration):
        # Make Data with Full Duration [query_start_time ~ query_end_time]
        
        start_time =duration['start_time']
        end_time = duration['end_time']
        # print("get_multipleDF_sameDuration", start_time, end_time)
        if len(data)>0:
            #2. Make Full Data(query Start ~ end) with NaN
            data.index = data.index.tz_localize(None) # 지역정보 없이 시간대만 가져오기
            new_idx = pd.date_range(start = start_time, end = (end_time- self.frequency), freq = self.frequency)
            new_data = pd.DataFrame(index= new_idx)
            new_data.index.name ='time' 
            data = new_data.join(data) # new_data에 data를 덮어쓴다
        return data

    """