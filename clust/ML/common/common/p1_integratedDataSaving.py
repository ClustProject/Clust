import os
import sys
import json
import datetime

sys.path.append("../../")
sys.path.append("../../..")
from Clust.clust.transformation.general.dataScaler import encode_hash_style
from Clust.clust.integration.utils import param

# 1. IntegratedDataSaving
# JH TODO 아래 코드에 대한 주석 작성
# JH TODO Influx Save Load 부분 작성 보완해야함

def getData(db_client, dataInfo, integration_freq_sec, process_param, startTime, endTime, integration_method = 'meta', method_param = {}, integration_duration_type = 'common'):
    intDataInfo = param.makeIntDataInfoSet(dataInfo, startTime, endTime)

    integration_param = getIntegrationParam(integration_freq_sec, integration_method, method_param, integration_duration_type)

    from Clust.app.integration_app1 import integration_from_influx_info
    data = integration_from_influx_info(db_client, intDataInfo, process_param, integration_param)


    return data

def integrated_data_meta(dataInfo, start_time, end_time, integration_freq_sec, clean_param, process_param):
    integrated_data_meta = {}
    integrated_data_meta["dataInfo"] = dataInfo
    integrated_data_meta["startTime"] = start_time
    integrated_data_meta["endTime"] = end_time
    integrated_data_meta["cleanParam"] = clean_param
    integrated_data_meta["integration_freq_sec"] = integration_freq_sec
    integrated_data_meta["process_param"] = process_param
    
    return integrated_data_meta

def getIntDataFromDataset(integration_freq_sec, processParam, multiple_dataset, integration_method = 'meta', method_param = {}, integration_duration_type = 'common'):
    integration_param = getIntegrationParam(integration_freq_sec, integration_method, method_param, integration_duration_type)
    ## Preprocessing
    from Clust.clust.preprocessing import processing_interface
    multiple_dataset = processing_interface.get_data_result('step_3', multiple_dataset, processParam)
    
    from Clust.clust.integration import integration_interface
    data = integration_interface.get_data_result('multiple_dataset_integration', multiple_dataset, integration_param)
            
    return data



# ============================== name 관련 ==============================
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



# ============================== parameter 관련 ==============================
def get_process_param(clean_param):
    if clean_param == "Clean":
        refine_param = {
            "remove_duplication": {"flag": True},"static_frequency": {"flag": True, "frequency": None}
        }
        CertainParam = {'flag': True}
        uncertainParam = {'flag': True, "param": {
            "outlierDetectorConfig": [
                {'algorithm': 'IQR', 'percentile': 99,
                    'alg_parameter': {'weight': 100}}
            ]}}
        outlier_param = {
            "certain_error_to_NaN": CertainParam,
            "uncertain_error_to_NaN": uncertainParam
        }
        imputation_param = {
            "flag": False,
            "imputation_method": [{"min": 0, "max": 3, "method": "linear", "parameter": {}}],
            "total_non_NaN_ratio": 80
        }

    else:
        refine_param = {"remove_duplication": {"flag": False},"static_frequency": {"flag": False, "frequency": None}}
        CertainParam = {'flag': False}
        uncertainParam = {'flag': False, "param": {}}
        outlier_param = {
            "certain_error_to_NaN": CertainParam,
            "uncertain_error_to_NaN": uncertainParam
        }
        imputation_param = {
                "flag": False,
                "imputation_method": [],
                "total_non_NaN_ratio": 80
        }

    process_param = {'refine_param': refine_param,
                     'outlier_param': outlier_param, 'imputation_param': imputation_param}
    return process_param


def getIntegrationParam(integration_freq_sec, integration_method, method_param, integration_duration_type):
    timedelta_frequency_min = datetime.timedelta(seconds= integration_freq_sec)
    integration_param = {
        "integration_frequency": timedelta_frequency_min,
        "integration_duration_type" : integration_duration_type,
        "param": method_param,
        "method": integration_method
    }
    
    return integration_param




# ============================== json 관련 ==============================
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
