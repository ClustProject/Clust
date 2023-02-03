import os
import sys
import json

sys.path.append("../../")
sys.path.append("../../..")
from Clust.clust.transformation.general.dataScaler import encode_hash_style

# 1. IntegratedDataSaving
# JH TODO 아래 코드에 대한 주석 작성
# JH TODO Influx Save Load 부분 작성 보완해야함

def get_list_merge(info_list):
    mergerd_name = ''
    for info in info_list:
        mergerd_name = mergerd_name+info+'_'
    return mergerd_name


def getNewDataName(process_param, data_info, integration_freq_sec, clean_param, data_save_mode, start_time, end_time):
    data_description_info = encode_hash_style(get_list_merge([str(process_param), str(data_info), str(integration_freq_sec), clean_param, data_save_mode]))
    time_interval_info = encode_hash_style(get_list_merge([start_time, end_time]))
    data_name = data_description_info+'_'+time_interval_info
    return data_name


def save_csv_data(data_folder_path, data_name, data):
    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)

    file_name = os.path.join(data_folder_path, data_name + '.csv')
    data.to_csv(file_name)
    return file_name


def save_influx_data(db_name, data_name, data, db_client):
    bk_name = db_name
    ms_name = data_name
    db_client.write_db(bk_name, ms_name, data)


def get_process_param(clean_param):
    if clean_param == "Clean":
        refine_param = {
            "removeDuplication": {"flag": True},
            "staticFrequency": {"flag": True, "frequency": None}
        }
        CertainParam = {'flag': True}
        uncertainParam = {'flag': True, "param": {
            "outlierDetectorConfig": [
                {'algorithm': 'IQR', 'percentile': 99,
                    'alg_parameter': {'weight': 100}}
            ]}}
        outlier_param = {
            "certainErrorToNaN": CertainParam,
            "unCertainErrorToNaN": uncertainParam
        }
        imputation_param = {
            "flag": False,
            "imputation_method": [{"min": 0, "max": 3, "method": "linear", "parameter": {}}],
            "totalNonNanRatio": 80
        }

    else:
        refine_param = {
            "removeDuplication": {"flag": False},
            "staticFrequency": {"flag": False, "frequency": None}
        }
        CertainParam = {'flag': False}
        uncertainParam = {'flag': False, "param": {}}
        outlier_param = {
            "certainErrorToNaN": CertainParam,
            "unCertainErrorToNaN": uncertainParam
        }
        imputation_param = {
                "flag": False,
                "imputation_method": [],
                "totalNonNanRatio": 80
        }

    process_param = {'refine_param': refine_param,
                     'outlier_param': outlier_param, 'imputation_param': imputation_param}
    return process_param


def getIntegrationParam(integration_freq_sec, integration_method, method_param, integration_duration):
    integration_param = {
        "integration_frequency": integration_freq_sec,
        "integration_duration" : integration_duration,
        "param": method_param,
        "method": integration_method
    }
    
    return integration_param


def getData(db_client, dataInfo, integration_freq_sec, processParam, startTime, endTime, integration_method = 'meta', method_param = {}, integration_duration = 'common'):
    from Clust.clust.integration.utils import param
    intDataInfo = param.makeIntDataInfoSet(dataInfo, startTime, endTime)

    integrationParam = getIntegrationParam(integration_freq_sec, integration_method, method_param, integration_duration)

    from Clust.clust.integration.integrationInterface import IntegrationInterface
    data = IntegrationInterface().integrationByInfluxInfo(db_client, intDataInfo, processParam, integrationParam)

    return data

def getIntDataFromDataset(integration_freq_sec, processParam, dataSet, integration_method = 'meta', method_param = {}, integration_duration = 'common'):
    integrationParam = getIntegrationParam(integration_freq_sec, integration_method, method_param, integration_duration)
    
    from Clust.clust.integration.integrationInterface import IntegrationInterface
    data = IntegrationInterface().multipleDatasetsIntegration(processParam, integrationParam, dataSet)

    return data


def writeJsonData(json_file_path, data):
    if os.path.isfile(json_file_path):
        pass
    else:
        directory = os.path.dirname(json_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(json_file_path, 'w') as f:
            json_data = {}
            json.dump(json_data, f, indent=2)
            print("New json file is created from data.json file")

    with open(json_file_path, 'w') as outfile:
        outfile.write(json.dumps(data, ensure_ascii=False))


def read_json_data(json_file_path):

    if os.path.isfile(json_file_path):
        pass
    else:
        directory = os.path.dirname(json_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(json_file_path, 'w') as f:
            data = {}
            json.dump(data, f, indent=2)
            print("New json file is created from data.json file")

    if os.path.isfile(json_file_path):
        with open(json_file_path, 'r') as json_file:
            json_data = json.load(json_file)
    return json_data


def saveJsonMeta(data_meta_path, data_name, process_param, dataInfo, integration_freq_sec, start_time, end_time, clean_param, data_save_mode):

    data_mata = read_json_data(data_meta_path)

    data_info = {}
    data_info["startTime"] = start_time
    data_info["endTime"] = end_time
    data_info["dataInfo"] = dataInfo
    data_info["processParam"] = process_param
    data_info["integration_freq_sec"] = integration_freq_sec
    data_info["cleanParam"] = clean_param
    data_info["DataSaveMode"] = data_save_mode
    data_mata[data_name] = {}
    data_mata[data_name]["integrationInfo"] = data_info

    writeJsonData(data_meta_path, data_mata)







# ----------------------------------------------------------------------------------------------
# new functions

def save_json_meta(mongodb_client, data_name, process_param, data_info, integration_freq_sec, start_time, end_time, clean_param, data_save_mode):

    meta_data = read_json_data(mongodb_client)

    meta_info = {}
    meta_info["startTime"] = start_time
    meta_info["endTime"] = end_time
    meta_info["dataInfo"] = data_info
    meta_info["processParam"] = process_param
    meta_info["integration_freq_sec"] = integration_freq_sec
    meta_info["cleanParam"] = clean_param
    meta_info["DataSaveMode"] = data_save_mode
    meta_data[data_name] = {}
    meta_data[data_name]["integrationInfo"] = meta_info

    save_meta_data(mongodb_client, meta_data)


def save_meta_data(mongodb_client, meta_data):
    db_name = 'integration'
    collection_name = 'meta'
    mongodb_client.insert_document(db_name, collection_name, meta_data)