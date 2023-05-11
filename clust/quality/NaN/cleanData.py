import sys
sys.path.append("../")
sys.path.append("../..")
from Clust.clust.quality.NaN import data_remove_byNaN

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

        DRN = data_remove_byNaN.DataRemoveByNaNStatus()
        nan_removed_data = DRN.removeNaNData(data, NanInfoForCleanData)

        print(len(data.columns), "--->", len(nan_removed_data.columns))

        return nan_removed_data
