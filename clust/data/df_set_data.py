import sys
sys.path.append("../")
sys.path.append("../../")

from Clust.clust.data import df_data
class DfSetData():
    def __init__(self, db_client):
        """
        # Description
         This class makes dataframe style data based on ingestion type, and param

        # Args
         * db_client (_str_): instance of InfluxClient class. instance to get data from influx DB

        """
        self.db_client = db_client
    
    def get_result(self, ingestion_type, ingestion_param, process_param = None):
        """
        # Description
         get dataframe result according to intestion_type, and ingestion_param

        # Args
         * ingestion_type (_str_)
         * ingestion_param (_dict_) : ingestion parameter depending on ingestion_type
         * process_param (_dict_, optional) : data preprocessing paramter. Defaults to None.

        # Returns
         * result (_pd.dataFrame_)

        """
        # define param
        self.ingestion_param    = ingestion_param
        self.process_param      = df_data.get_default_process_param(process_param)
        
        # get result
        if ingestion_type == "multi_numeric_ms_list": # general type
            result = self.multi_numeric_ms_list(self.ingestion_param)
            
        elif ingestion_type == "multi_ms_one_enumerated_ms_in_bucket_integration":
            result = self.multi_ms_one_enumerated_ms_in_bucket_integration(self.ingestion_param, self.process_param)
            
        elif ingestion_type == "all_ms_in_one_bucket":
            result = self.all_ms_in_one_bucket(self.ingestion_param)
        
        elif ingestion_type == "all_ms_in_multiple_bucket":
            result = self.all_ms_in_multiple_bucket(self.ingestion_param)
            
        return result
    
    def multi_numeric_ms_list(self, ingestion_param):
        """
        # Description
         Get only numeric ms data list by ingestion_param without any preprocessing
        
        # Args
         * ingestion_param (_Dict_) : intDataInfo or ingestion_param        
        ```
        >>> ingestion_param = {
                                'start_time': '2021-09-05 00:00:00', 
                                'end_time': '2021-09-11 00:00:00', 
                                'feature_list': ['CO2', 'out_PM25'], 
                                'ms_list_info': [['air_outdoor_kweather', 'OC3CL200012'], 
                                                    ['air_outdoor_keti_clean', 'seoul'], 
                                                    ['air_indoor_modelSchool', 'ICW0W2000011']]
                            }
        ```

        # Returns
         * MSdataset (_dict_)

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
        for dbinfo in intDataInfo['db_info']:
            db_name     = dbinfo['db_name']
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
                MSdataSet[data_name]  =  multiple_dataset.select_dtypes(include=np.number)
                MSdataSet[data_name].index.name ='datetime'

        return MSdataSet
    
    def multi_ms_one_enumerated_ms_in_bucket_integration(self, ingestion_param, process_param) :
        """
        # Description
         1개의 특정 bucket에 있는 모든 ms (multiple ms in bucket) 와 고정된 다른 ms (ms_list_info) 들을 복합하여 데이터를 준비함.
         feature_list 가 명시 되었다면, 명시된 feature_list와 관련한 데이터만 전달.

        # Args
         * ingestion_param (_dict_) : ingestion_param 
        ```
        >>> ingestion_param = {'bucket_name': 'air_indoor_modelSchool', 
                            'data_org': [['air_outdoor_kweather', 'OC3CL200012'], ['air_outdoor_keti_clean', 'seoul']], 
                            'start_time': '2021-09-05 00:00:00', 
                            'end_time': '2021-09-11 00:00:00', 
                            'integration_freq_min': 60, 
                            'feature_list': ['CO2', 'out_PM10', 'out_PM25']}
        ```
         * process_param (_dict_) : preprocessing parameter
        
        # Returns
         * bucket_dataSet (_dict_) : integrated data

        """

        data_org            = ingestion_param['data_org']
        bucket_name         = ingestion_param['bucket_name']
        start_time          = ingestion_param['start_time']     #현재 사용되지 않는 파라미터, 삭제 해야하는지 추후 확인
        end_time            = ingestion_param['end_time']   #현재 사용되지 않는 파라미터, 삭제 해야하는지 추후 확인
        integration_freq_sec = int(ingestion_param['integration_freq_min']) * 60    #현재 사용되지 않는 파라미터, 삭제 해야하는지 추후 확인
        
        bucket_dataSet = {}        
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
    
    def all_ms_in_one_bucket(self, ingestion_param):
        """
        # Description
         It returns dataSet from all MS of a speicific DB(Bucket) from start_time to end_time

        # Args
         * ingestion_param(_dict_) : bucket_name, start_time, end_time  + feature_list(optional)
        
        # Returns
         * dataSet (_dict_) : returned dataset ----> {"data1_name":DF1, "data2_name:DF2......}

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
        # Description
         get all ms dataset in bucket_list (duration: start_time ~ end_time).
         if new_bucket_list is not None, change bucket_list name.

        # Args
         * ingestion_param (_dict_) : array of multiple bucket name 
        
        # Returns
         * data_set (_dict of pd.DataFrame_) : new DataSet : key name  ===> msName + bucketName

        """

        bucket_list = ingestion_param['bucket_list']
        start_time  = ingestion_param['start_time'] #현재 사용되지 않는 파라미터, 삭제 해야하는지 추후 확인
        end_time    = ingestion_param['end_time'] #현재 사용되지 않는 파라미터, 삭제 해야하는지 추후 확인
        
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

            dataSet_indi = {f'{k}_{new_bucket_list[idx]}': v for k, v in dataSet_indi.items()}
            data_set.update(dataSet_indi)

        return data_set



