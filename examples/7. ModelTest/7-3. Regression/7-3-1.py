import sys, os
import pandas as pd

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../../")

from Clust.clust.ML.common.common import p1_integratedDataSaving as p1

from Clust.setting import influx_setting_KETI as ins
from Clust.clust.ingestion.influx import influx_client_v2 as influx_Client
from Clust.clust.ingestion.mongo.mongo_client import MongoClient

db_client = influx_Client.InfluxClient(ins.CLUSTDataServer2)
mongo_client = MongoClient(ins.CLUSTMetaInfo2)

##### #0. Common###########################################################################################
clean_param_list=['Clean', 'NoClean']
bucket_name = 'integration_info'

def make_data_set(ingestion_param, clean_param, process_param, integration_param):
    
    from Clust.clust.data import data_interface
    multiple_dataset = data_interface.get_data_result("multiple_ms_by_time", db_client, ingestion_param)

    # 2. Data Preprocessing
    from Clust.clust.preprocessing import processing_interface
    multiple_dataset = processing_interface.get_data_result('step_3', multiple_dataset, process_param)

    # 3. Data Integration
    from Clust.clust.integration.integrationInterface import IntegrationInterface
    
    data = IntegrationInterface().multipleDatasetsIntegration(integration_param, multiple_dataset)
    
    
    data = IntegrationInterface().multipleDatasetsIntegration(integration_param, multiple_dataset)
        
    return data

##### #1. For Regression######################################################################################################################################
data_mode_list =['trainX', 'trainy', 'testX', 'testy']
trainStartTime = "2016-01-11"
trainEndTime = "2016-04-15"
testStartTime = "2021-01-01"
testEndTime = "2021-02-12"

for clean_param in clean_param_list:
    process_param = p1.get_process_param(clean_param)
    for data_mode in data_mode_list:
        data_name_pre ='energy_regression_'
        ingestion_param ={}
        integration_param = {
            #"integration_frequency":integration_freq_sec,
            "param":{},
            "method":"meta",
            "integration_duration":"common"
        }
        if data_mode == 'trainX':
            ingestion_param['start_time'] = trainStartTime
            ingestion_param['end_time'] = trainEndTime
            ingestion_param['ms_list_info'] = [['life_indoor_environment', 'humidityTrain_10min'], 
                        ['life_indoor_environment', 'temperatureTrain_10min'], 
                        ['weather_outdoor_environment', 'belgiumChieverseAirportTrain_10min']]
            integration_param["integration_frequency"] = 60 * 10 # 10분

        elif data_mode =='trainy':
            ingestion_param['start_time'] = testStartTime
            ingestion_param['end_time'] = testEndTime
            ingestion_param['ms_list_info'] = [['life_indoor_environment', 'humidityTest_10min'], 
                        ['life_indoor_environment', 'temperatureTest_10min'], 
                        ['weather_outdoor_environment', 'belgiumChieverseAirportTest_10min']]
            integration_param["integration_frequency"]= 60 * 10 # 10분

        elif data_mode == 'testX':
            ingestion_param['start_time'] = trainStartTime
            ingestion_param['end_time'] = trainEndTime
            ingestion_param['ms_list_info'] = [['life_indoor_environment', 'applianceEnergyDatasetTrainy_1day']]
            integration_param["integration_frequency"] = 60 * 60 * 24 # 24시간

        elif data_mode =='testy':
            ingestion_param['start_time'] = testStartTime
            ingestion_param['end_time'] = testEndTime
            ingestion_param['ms_list_info'] = [['life_indoor_environment', 'applianceEnergyDatasetTesty_1day']]
            integration_param["integration_frequency"] = 60 * 60 * 24 # 24시간
            
        # dataset ingestion---> data preprocessing ---> data integration ---> information save
        # 1. Ingestion multiple dataset

        data = make_data_set(ingestion_param, clean_param, process_param, integration_param)
        ms_name = data_name_pre + data_mode + clean_param
        # save to influxdb
        db_client.write_db(bucket_name, ms_name, data)
        # save to mongodb
        meta_info={"ingestion_param": ingestion_param, "integration_param":integration_param, "clean_param":clean_param, "process_param":process_param}
        mongo_client.insert_document(bucket_name, ms_name, meta_info)
        print(meta_info)

##### #2. For Forecasting##################################################################################################################################   

data_name_list=['Hs1SwineFarmWithWeatherTime', 'gunwiStrawberryWithWeatherTime', 'strawberryOpenTime']
integration_freq_sec = 60 * 5
data_mode_list=["train", "test"]
for clean_param in clean_param_list:
    ingestion_param={}
    process_param = p1.get_process_param(clean_param)
    integration_param['integration_freq_sec'] = integration_freq_sec
    integration_param = {
            #"integration_frequency":integration_freq_sec,
                "param":{},
                "method":"meta",
                "integration_duration":"common"
    }
    for data_name in data_name_list:
        for data_mode in data_mode_list:
            if data_name==data_name_list[0]:
                if data_mode == 'train':
                    ingestion_param['start_time'] = "2021-02-01 00:00:00"
                    ingestion_param['end_time']  ="2021-03-10 00:00:00"
                elif data_mode =='test':
                    ingestion_param['start_time']  ="2021-03-10 00:00:00"
                    ingestion_param['end_time']  ="2021-03-17 00:00:00"
                ingestion_param['ms_list_info'] = [['farm_swine_air', 'HS2'], ['weather_outdoor_keti_clean', 'sangju'], ['life_additional_Info', 'trigonometicInfoByHours']]

            elif data_name==data_name_list[1]:
                    if data_mode == 'train':
                        ingestion_param['start_time'] = "2022-01-22 00:00:00"
                        ingestion_param['end_time']  ="2022-02-25 00:00:00"
                    elif data_mode =='test':
                        ingestion_param['start_time']  ="2022-02-25 00:00:00"
                        ingestion_param['end_time']  ="2022-02-28 00:00:00"
                    ingestion_param['ms_list_info'] = [['farm_strawberry_gunwi', 'control_environment'], ['farm_strawberry_gunwi', 'environment'], ['life_additional_Info', 'trigonometicInfoByHours']]
                    

            elif data_name ==data_name_list[2]:
                if data_mode == 'train':
                        ingestion_param['start_time'] = "2022-01-22 00:00:00"
                        ingestion_param['end_time']  ="2022-02-25 00:00:00"
                elif data_mode =='test':
                    ingestion_param['start_time']  ="2022-02-25 00:00:00"
                    ingestion_param['end_time']  ="2022-02-28 00:00:00"
                ingestion_param['ms_list_info'] = [['farm_strawberry_gunwi', 'control_environment'], ['farm_strawberry_gunwi', 'environment'], ['life_additional_Info', 'trigonometicInfoByHours']]
                    
            
            make_data_set(ingestion_param, clean_param, process_param, integration_param)
            ms_name = data_name+'_' + data_mode+'_' + clean_param
            # save to influxdb
            db_client.write_db(bucket_name, ms_name, data)
            # save to mongodb
            meta_info={"ingestion_param": ingestion_param, "integration_param":integration_param, "clean_param":clean_param, "process_param":process_param}
            mongo_client.insert_document(bucket_name, ms_name, meta_info)
            print(meta_info)
            