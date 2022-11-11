
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from KETIPreDataIngestion.dataByCondition.cycleData import CycleData


def getCycleselectDataFrame(query_data, feature_cycle, feature_cycle_times, frequency=None):
    from KETIPrePartialDataPreprocessing.data_preprocessing import DataPreprocessing
    refine_param = {'removeDuplication': {'flag': True}, 'staticFrequency': {'flag': True, 'frequency': frequency}}
    output_data = DataPreprocessing().get_refinedData(query_data, refine_param)
    cycleData = CycleData()
    # cycle 주기에 따라 적절한 함수 적용
    if feature_cycle == 'Hour':
        data = cycleData.getHourCycleSet(output_data, feature_cycle_times, False)
    elif feature_cycle == 'Day':
        data = cycleData.getDayCycleSet(output_data, feature_cycle_times, False)
    elif feature_cycle == 'Week':
        data = cycleData.getWeekCycleSet(output_data, feature_cycle_times, False)
    elif feature_cycle == 'Month':
        data = cycleData.getMonthCycleSet(output_data, feature_cycle_times, False)
    elif feature_cycle == 'Year':
        data = cycleData.getYearCycleSet(output_data, feature_cycle_times, False)

    return data


def getCycleSelectDataSet(query_data, feature_cycle, feature_cycle_times, frequency=None):
    data_list = getCycleselectDataFrame(query_data, feature_cycle, feature_cycle_times, frequency)
    dataSet = {}
    for data_slice in data_list:
        index_name = data_slice.index[0].strftime('%Y-%m-%d %H:%M:%S')
        dataSet[index_name] = data_slice

    return dataSet



















# if __name__ == '__main__':
#     from KETIPreDataIngestion.KETI_setting import influx_setting_KETI as ins
#     from KETIPreDataIngestion.data_influx.influx_Client import influxClient
#     import pandas as pd

#     db_client = influxClient(ins.CLUSTDataServer2)
#     query_start_time = pd.to_datetime("2021-02-05 00:00:00")
#     query_end_time = pd.to_datetime("2021-03-05 00:00:00")

#     durationList = ['1 hour', '1 day', '1 week', '1 month','1 year']
#     feature_cycle = '1 day'
#     feature_cycle_times = 1

#     modelName="som"
#     db_name = 'air_indoor_경로당'
#     ms_name = "ICL1L2000238"
#     feature_list = ['in_temp']
#     feature_name = feature_list[0]
#     freq_min = 60

#     NanInfoForClenData = {'type':'num', 'ConsecutiveNanLimit':1, 'totalNaNLimit':10}

#     bind_params = {'end_time':query_end_time.strftime('%Y-%m-%dT%H:%M:%S'), 'start_time': query_start_time.strftime('%Y-%m-%dT%H:%M:%S')}


#     test = db_client.get_data_by_time(start_time, end_time, db_name, ms_name)

#     from KETIPrePartialDataPreprocessing.data_preprocessing import DataPreprocessing
#     refine_param = {'removeDuplication': {'flag': True}, 'staticFrequency': {'flag': True, 'frequency': None}}
#     output_data = DataPreprocessing().get_refinedData(test, refine_param)

#     data = getCycleselectDataFrame(output_data, feature_cycle, feature_cycle_times)
#     # print(data)

