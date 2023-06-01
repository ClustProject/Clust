import sys
import datetime
sys.path.append("../")
sys.path.append("../../")

class DfSetData():
    def __init__(self, db_client):
        """
        This class makes dataframe style data based on ingestion type, and param.

        Args:
            db_client (_instance_): Instance of InfluxClient class. Instance to get data from influx DB.
        """
        self.db_client = db_client
    
    def get_result(self, ingestion_type, ingestion_param):
        """
        Get dictionary result with dataframe as value according to ingestion_type, and ingestion_param.

        Args:
            ingestion_type (String): ingestion_type (method)
            ingestion_param (Dictionary): ingestion parameter depending on ingestion_type

        Returns:
            result (_pd.dataFrame_)
        """
        # define param
        self.ingestion_param    = ingestion_param
        
        # get result
        if ingestion_type == "multiple_ms_by_time": # general type
            result = self.multiple_ms_by_time(self.ingestion_param)
            
        elif ingestion_type == "multi_ms_one_enumerated_ms_in_bucket_integration":
            result = self.multi_ms_one_enumerated_ms_in_bucket_integration(self.ingestion_param)
            
        elif ingestion_type == "all_ms_in_one_bucket":
            result = self.all_ms_in_one_bucket(self.ingestion_param)
        
        elif ingestion_type == "all_ms_in_multiple_bucket":
            result = self.all_ms_in_multiple_bucket(self.ingestion_param)
            
        return result
    
    def multiple_ms_by_time(self, ingestion_param):
        """
        Collect data set(type is dictionary) whose values are dataframe with only numeric values collected by time duration
        The parameter "feature_list" is designated for each data.
        
        Args:
            ingestion_param (Dictionary) : intDataInfo or ingestion_param        
        
        Returns:
            Dictionary: MSdataset
        
        Example:
            >>> ingestion_param = {
            ...        'start_time': '2022-11-01 00:00:00', 
            ...        'end_time': '2022-11-08 00:00:00', 
            ...        'feature_list': [['in_co2', 'in_humi'], ['in_co2', 'in_noise'], ['in_co2', 'in_temp']], 
            ...        'ms_list_info': [['air_indoor_중학교', 'ICW0W2000010'], 
            ...                        ['air_indoor_초등학교', 'ICW0W2000034'], 
            ...                        ['air_indoor_도서관', 'ICW0W2000094']]
            ...        }

        """       
        #-------------------------------------------------------------------------------------------------------------------------------
        # 여기서 ingestion_param이 순수 ingestion param일 수도 있고 intDataInfo 형태일 수도 있어서.. 모두 수정하기가 어려워 아래에 관련한 부분에 대한 처리 코드를 임시적으로 넣었음
        if 'ms_list_info' in ingestion_param.keys():
            ms_list_info        = ingestion_param['ms_list_info']
            start_time          = ingestion_param['start_time']
            end_time            = ingestion_param['end_time']
            from Clust.clust.integration.utils import param
            intDataInfo = param.makeIntDataInfoSet(ms_list_info, start_time, end_time) 
        else:
            intDataInfo = ingestion_param
        #-------------------------------------------------------------------------------------------------------------------------------
        
        MSdataSet ={}
        for idx, dbinfo in enumerate(intDataInfo['db_info']):
            db_name     = dbinfo["bucket_name"]
            ms_name     = dbinfo['measurement']
            data_name   = db_name + "_" + ms_name
            tag_key     = None
            tag_value   = None 

            if "tag_key" in dbinfo.keys():
                if "tag_value" in dbinfo.keys():
                    tag_key = dbinfo['tag_key']
                    tag_value = dbinfo['tag_value']

            import numpy as np
            multiple_dataset=self.db_client.get_data_by_time(dbinfo['start'], dbinfo['end'], db_name, ms_name, tag_key, tag_value)
            if not(multiple_dataset.empty):
                if 'feature_list' in ingestion_param.keys():
                    feature_list= ingestion_param['feature_list'][idx]
                    multiple_dataset = multiple_dataset[feature_list]
                MSdataSet[data_name]  =  multiple_dataset.select_dtypes(include=np.number)
                MSdataSet[data_name].index.name ='datetime'

        return MSdataSet
    
    def multi_ms_one_enumerated_ms_in_bucket_integration(self, ingestion_param) :
        """
        1개의 특정 bucket에 있는 모든 ms (multiple ms in bucket) 와 고정된 다른 ms (ms_list_info) 들을 복합하여 데이터를 준비함.
        feature_list 가 명시 되었다면, 명시된 feature_list와 관련한 데이터만 전달.
        feature_list는 list of list 형태로 각 데이터 별 지정.

        Args:
            ingestion_param (Dictionary) : ingestion_param 
        
        Returns:
            Dictionary: integrated data # bucket_dataSet

        Example:
            >>> ingestion_param = {
            ...         'bucket_name': 'air_indoor_체육시설', 
            ...         'data_org': [['air_outdoor_kweather', 'OC3CL200012'], ['air_outdoor_keti_clean', 'seoul']], 
            ...         'start_time': '2021-09-05 00:00:00', 
            ...         'end_time': '2021-09-11 00:00:00', 
            ...         'integration_freq_min': 60, 
            ...         'feature_list': [['out_pm10', 'out_pm25'], ['out_PM10', 'out_PM25'], ['in_pm01', 'in_pm10', 'in_pm25']]}
        """

        data_org            = ingestion_param['data_org']
        bucket_name         = ingestion_param['bucket_name']
        
        bucket_dataSet = {}        
        ms_list = self.db_client.measurement_list(bucket_name) #ms_name
        for ms_name in ms_list:
            dataInfo = data_org
            dataInfo = data_org + [[bucket_name, ms_name]] 
            ingestion_param['ms_list_info'] = dataInfo
            
            ############ ingestion
            from Clust.clust.data import data_interface
            multiple_dataset = data_interface.get_data_result("multiple_ms_by_time", self.db_client, ingestion_param)
            ############ Preprocessing
            from Clust.clust.preprocessing import processing_interface
            multiple_dataset = processing_interface.get_data_result('step_3', multiple_dataset)
            #############
            # data Integration
            integration_freq_min = datetime.timedelta(minutes = ingestion_param['integration_freq_min'])
            integration_param   = {
                "integration_duration_type":"common",
                "integration_frequency":integration_freq_min,
                "param":{},
                "method":"meta"
            }
            
            from Clust.clust.integration import integration_interface
            dataIntegrated = integration_interface.get_data_result('multiple_dataset_integration', multiple_dataset, integration_param)

            ############
            bucket_dataSet[ms_name]= dataIntegrated
        
        return bucket_dataSet
    
    def all_ms_in_one_bucket(self, ingestion_param):
        """
        It returns dataSet from all MS of a speicific DB(Bucket) from start_time to end_time

        Args:
            ingestion_param(Dictionary) : ingestion param (bucket_name, start_time, end_time  + feature_list(optional))
        
        Returns:
            Dictionary: returned dataset ----> {"data1_name":DF1, "data2_name:DF2......}

        Example: 
            >>> ingestion_param = {
            ...    'bucket_name' : 'air_indoor_modelSchool', 
            ...    'start_time': '2021-09-05 00:00:00',
            ...    'end_time': '2021-09-11 00:00:00',
            ...    'feature_list' : [ 'CO2', 'Noise','PM10','PM25', 'Temp', 'VoCs', 'humid' ]}

        """

        bucket_name     = ingestion_param['bucket_name']
        start_time      = ingestion_param['start_time'] 
        end_time        = ingestion_param['end_time']

        ms_list = self.db_client.measurement_list(bucket_name)
        dataSet ={}
        for ms_name in ms_list:
            data = self.db_client.get_data_by_time(start_time, end_time, bucket_name, ms_name)
            if len(data)>0:
                if 'feature_list'in ingestion_param.keys():
                    feature_list= ingestion_param['feature_list'] 
                    data = data[feature_list]
                dataSet[ms_name] = data

        return dataSet

    def all_ms_in_multiple_bucket(self, ingestion_param):    
        """        
        Get all ms dataset in bucket_list (duration: start_time ~ end_time).
        If new_bucket_list is not None, change bucket_list name.

        Args:
            ingestion_param (Dictionary) : array of multiple bucket name 
        
        Returns:
            Dictionary : new DataSet (key name  ===> msName + bucketName)

        Example:
            >>> ingestion_param = {
            ...     'bucket_list' : ['air_indoor_중학교', 'air_indoor_도서관'],
            ...     'new_bucket_list' : ['library', 'middleSchool'],
            ...     'start_time' : '2022-11-01 00:00:00',
            ...     'end_time' : '2022-11-08 00:00:00',
            ...     'feature_list' : ['in_co2', 'in_humi', 'in_noise', 'in_temp']
            }
        """
        bucket_list = ingestion_param['bucket_list']
        
        if 'new_bucket_list' in ingestion_param.keys():
            new_bucket_list = ingestion_param['new_bucket_list']
        else:
            new_bucket_list = bucket_list

        data_set = {}
        for idx, bucket_name in enumerate(bucket_list):
            # data exploration start
            ingestion_param['bucket_name'] = bucket_name
            dataSet_indi = self.all_ms_in_one_bucket(ingestion_param)

            print(bucket_name, " length:", len(dataSet_indi))

            dataSet_indi = {f'{k}/{new_bucket_list[idx]}': v for k, v in dataSet_indi.items()}
            data_set.update(dataSet_indi)

        return data_set
