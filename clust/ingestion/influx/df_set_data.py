import sys
sys.path.append("../")
sys.path.append("../../")

from Clust.clust.ingestion.influx import df_data
class DfSetData():
    def __init__(self, db_client):
        """This class makes dataframe style data based on ingestion type, and param

        Args:
            db_client (string): db_client (instance of influxClient class): instance to get data from influx DB
        """
        self.db_client = db_client
    
    def get_result(self, ingestion_type, ingestion_param, process_param = None):
        """get dataframe result according to intestion_type, and ingestion_param

        Args:
            ingestion_type (string): ingestion_type (method)
            ingestion_param (dictionary): ingestion parameter depending on ingestion_type
            process_param (dictionary, optional): data preprocessing paramter. Defaults to None.

        Returns:
            dataframe: result
        """
        # define param
        self.ingestion_param = ingestion_param
        self.process_param = df_data.check_process_param(process_param)
        
        # get result
        if ingestion_type == "multi_numeric_ms_list": # general type
            result = self.multi_numeric_ms_list(self.ingestion_param)
            
        elif ingestion_type == "multi_ms_one_enumerated_ms_in_bucket_integration":
            result = self.multi_ms_one_enumerated_ms_in_bucket_integration(self.ingestion_param, self.process_param)
            
        return result
    
    def multi_numeric_ms_list(self, ingestion_param):
        """
        Get only numeric ms data list by ingestion_param without any preprocessing
        Args:    
            ingestion_param (dict): intDataInfo or ingestion_param
        
        Returns:
            Dictionary: MSdataset
        """
        # 여기서 ingestion_param이 순수 ingestion param일 수도 있고 결합된 형태일 수도 있어서.. 모두 수정하기가 어려워 아래에 관련한 부분에 대한 처리 코드를 임시적으로 넣었음
        if 'ms_list_info' in ingestion_param.keys():
            ms_list_info        = ingestion_param['ms_list_info']
            start_time          = ingestion_param['start_time']
            end_time            = ingestion_param['end_time']
            from Clust.clust.integration.utils import param
            intDataInfo = param.makeIntDataInfoSet(ms_list_info, start_time, end_time) 
        else:
            intDataInfo = ingestion_param
        ##############################################################################################################################################
        
        MSdataSet ={}
        for i, dbinfo in enumerate(intDataInfo['db_info']):
            db_name = dbinfo['db_name']
            ms_name = dbinfo['measurement']
            tag_key =None
            tag_value =None 
            if "tag_key" in dbinfo.keys():
                if "tag_value" in dbinfo.keys():
                    tag_key = dbinfo['tag_key']
                    tag_value = dbinfo['tag_value']

            import numpy as np
            multiple_dataset=self.db_client.get_data_by_time(dbinfo['start'], dbinfo['end'], db_name, ms_name, tag_key, tag_value)
            MSdataSet[i]  =  multiple_dataset.select_dtypes(include=np.number)
            MSdataSet[i].index.name ='datetime'

        return MSdataSet
    
    def multi_ms_one_enumerated_ms_in_bucket_integration(self, ingestion_param, process_param):
        """1개의 특정 bucket에 있는 모든 ms (multiple ms in bucket) 와 고정된 다른 ms (ms_list_info) 들을 복합하여 데이터를 준비함
        feature_list 가 명시되었다면 명시된 feature_list와 관련한 데이터만 전달

        Args:
            ingestion_param (dict): ingestion_param 
            process_param (dict): preprocessing parameter

        >>> ingestion_param = {
            'data_org':,
            'bucket_name': , 
            'start_time':,
            'end_time':,
            'integration_freq_min:,
            'feature_list:[]
        }
        
        Returns:
            dictionary: integrated data
        """
        
        data_org        = ingestion_param['data_org']
        bucket_name         = ingestion_param['bucket_name']
        start_time          = ingestion_param['start_time']
        end_time            = ingestion_param['end_time']
        integration_freq_sec = int(ingestion_param['integration_freq_min']) * 60 
        bucket_dataSet={}
        
        ms_list = self.db_client.measurement_list(bucket_name) #ms_name
        for ms_name in ms_list:
            dataInfo = data_org
            dataInfo = data_org + [[bucket_name, ms_name]] 
            ingestion_param['ms_list_info'] = dataInfo
            dataIntegrated = df_data.DfData(self.db_client).get_result("multi_ms_integration", ingestion_param, process_param)
            if ingestion_param['feature_list']:
                dataIntegrated = dataIntegrated[ingestion_param['feature_list']]
            bucket_dataSet[ms_name]= dataIntegrated
        
        return bucket_dataSet



