import sys, pandas
sys.path.append("../")
sys.path.append("../../")

from Clust.clust.data import df_set_data

class DfData():
    def __init__(self, db_client):
        """
        # Description
        This class makes dataframe style data based on ingestion type, and param

        # Args
         * db_client (_instance_): influx db instance

        """
        self.db_client = db_client
        
    def get_result(self, ingestion_type, ingestion_param, process_param = None) -> pandas.DataFrame :
        """       
        # Description
         get dataframe result according to intestion_type, and ingestion_param

        # Args
         * ingestion_type (_str_): ingestion_type (method)
         ```
            >>> ingestion_type = ['multi_ms_integration', 'ms_by_days', 'ms_by_time', 'ms_by_num']  
         ```
         * ingestion_param (_Dict_): ingestion parameter depending on ingestion_type
         ```example

            >>> ingestion_param = {
                                    'integration_freq_min': '60', 
                                    'start_time': '2021-09-05 00:00:00', 
                                    'end_time': '2021-09-11 00:00:00', 
                                    'ms_list_info': [['air_indoor_modelSchool', 'ICW0W2000014'], 
                                                    ['air_outdoor_kweather', 'OC3CL200012'], 
                                                    ['air_outdoor_keti_clean', 'seoul']],
                                    'feature_list' : ['CO2', 'Noise', 'PM10', 'PM25', 'Temp', 'VoCs', 'humid',
                                                        'out_h2s','out_humi', 'out_noise', 'out_temp',
                                                    'out_ultraviolet_rays', 'out_PM10','out_PM25']                                
                                    }
                              
         ```
         * process_param (_Dict_, _optional_): data preprocessing paramter. Defaults to None.

        # Returns
         * result (_dataframe_)

        """

        # define param
        self.ingestion_param    = ingestion_param
        self.process_param      = get_default_process_param(process_param)
        
        # get result
        if ingestion_type == "multi_ms_integration":
            result = self.multi_ms_integration(self.ingestion_param)  
        elif ingestion_type == "ms_by_num":
            result = self.ms_by_num(self.ingestion_param) 
        elif ingestion_type == "ms_by_days":
            result = self.ms_by_days(self.ingestion_param) 
        elif ingestion_type == "ms_by_time":
            result = self.ms_by_time(self.ingestion_param) 
        
        if self.ingestion_param['feature_list']:
            if len(result)>0:
                result = result[self.ingestion_param['feature_list']]
            
        return result
    
    def ms_by_days(self, ingestion_param):
        """
        # Description
         data by days 

        # Args
         * ingestion_param (_dict_) 

        ```         
         >>> ingestion_param = {
                                    'days' : 1,                                     
                                    'end_time': '2021-09-11 00:00:00', 
                                    'db_name' : 'air_indoor_modelSchool',
                                    'ms_name' : 'ICW0W2000014',
                                    'feature_list' : ['CO2', 'Noise', 'PM10', 'PM25', 'Temp', 'VoCs', 'humid',
                                                        'out_h2s','out_humi', 'out_noise', 'out_temp',
                                                    'out_ultraviolet_rays', 'out_PM10','out_PM25']                                
                                    }
        ```
        
        # Returns
         * data (_pd.dataFrame_) : result data

        """
        #TODO: ingestion_param 의 bucket은 list안에 들어있다. for문 사용하는 소스로 개선할 것인지 확인 필요.
        data = self.db_client.get_data_by_days(ingestion_param['end_time'], ingestion_param['days'], ingestion_param['db_name'], ingestion_param['ms_name']) 

        return data
    
    def ms_by_time(self, ingestion_param):
        """
        # Description
         data by time duration

        # Args
         * ingestion_param (_dict_) 
        ```            
         >>> ingestion_param = {
                                'db_name' : 'air_indoor_modelSchool'
                                , 'ms_name' : 'ICW0W2000014'
                                , 'start_time': '2021-09-05 00:00:00'
                                , 'end_time': '2021-09-11 00:00:00'
                                , 'feature_list' : [ 'CO2', 'Noise','PM10','PM25', 'Temp', 'VoCs', 'humid' ]

                            }
        ```

        # Returns
         * data (_pd.dataFrame_) : result data

        """
        #TODO: ingestion_param 의 bucket은 list안에 들어있다. for문 사용하는 소스로 개선할 것인지 확인 필요.
        data = self.db_client.get_data_by_time(ingestion_param['start_time'], ingestion_param['end_time'], ingestion_param['db_name'], ingestion_param['ms_name'])
        
        return data
    
    def ms_by_num(self, ingestion_param):
        """
        # Description
         data by num 

        # Args
         * ingestion_param (_dict_)
        ```
        >>> ingestion_param = {
                                'db_name' : 'air_indoor_modelSchool'
                                , 'ms_name' : 'ICW0W2000014'
                                , 'num' : 1 #ms_by_num
                                , 'position' : 'end'
                                , 'feature_list' : [ 'CO2', 'Noise','PM10','PM25', 'Temp', 'VoCs', 'humid' ]

                            }
        ```

        # Returns
         * data (_pd.dataFrame_) : result data

        """
        #TODO: ingestion_param 의 bucket은 list안에 들어있다. for문 사용하는 소스로 개선할 것인지 확인 필요.
        if ingestion_param['position']=='end':
            data = self.db_client.get_data_end_by_num(ingestion_param['num'], ingestion_param['db_name'], ingestion_param['ms_name']) 
        else:
            data = self.db_client.get_data_front_by_num(ingestion_param['num'], ingestion_param['db_name'], ingestion_param['ms_name']) 
            
        return data 
    
    def multi_ms_integration(self, ingestion_param):
        """ 
        # Description
         get integrated numeric data with multiple MS data integration : ms1+ms2+....+msN => DF

        # Args
         * ingestion_param (_dict_) 
         ```
            >>> ingestion_param = {
                                    'integration_freq_min': '60', 
                                    'start_time': '2021-09-05 00:00:00', 
                                    'end_time': '2021-09-11 00:00:00', 
                                    'ms_list_info': [['air_indoor_modelSchool', 'ICW0W2000014'], 
                                                    ['air_outdoor_kweather', 'OC3CL200012'], 
                                                    ['air_outdoor_keti_clean', 'seoul']],
                                    'feature_list' : ['CO2', 'Noise', 'PM10', 'PM25', 'Temp', 'VoCs', 'humid',
                                                        'out_h2s','out_humi', 'out_noise', 'out_temp',
                                                    'out_ultraviolet_rays', 'out_PM10','out_PM25']                                
                                    }
         ```

        # Returns
         * integrated data (_pd.dataFrame_)
            
        """
        
        integration_freq_sec    = int(ingestion_param['integration_freq_min']) * 60 
        integration_param       = get_integration_param(integration_freq_sec)
        multiple_dataset        = df_set_data.DfSetData(self.db_client).get_result("multi_numeric_ms_list", ingestion_param)

        # data Integration
        from Clust.clust.integration.integrationInterface import IntegrationInterface
        dataIntegrated = IntegrationInterface().multipleDatasetsIntegration(self.process_param, integration_param, multiple_dataset)
        
        return dataIntegrated
            
            
def get_default_process_param(process_param):
    """
    # Description
     Check and generate general preprocessing parameter (if not exists)
    
    # Args
     * process_param (_dict_ or _None_) : input process param

    # Returns
     * process_param (_dict_)
    
    """
    refine_param        = {"removeDuplication":{"flag":True},"staticFrequency":{"flag":True, "frequency":None}}
    CertainParam        = {'flag': True}
    uncertainParam      = {'flag': False, "param": {"outlierDetectorConfig":[{'algorithm': 'IQR', 'percentile':99 ,'alg_parameter': {'weight':100}}]}}
    outlier_param       = {"certainErrorToNaN":CertainParam, "unCertainErrorToNaN":uncertainParam}
    imputation_param    = {
        "flag":True,
        "imputation_method":[{"min":0,"max":3,"method":"linear", "parameter":{}}],
        "totalNonNanRatio":80
    }

    # 최종 파라미터
    if process_param:
        process_param = process_param
    else:
        process_param = {'refine_param':refine_param, 'outlier_param':outlier_param, 'imputation_param':imputation_param}
    
    return process_param

def get_integration_param(integration_freq_sec) -> dict : 
    """
    # Description
     Generate general integration parameter

    # Args
     * integration_freq_sec (_int_)

    # Returns
     * integration_param (_dict_)

    """
    integration_param   = {
        "integration_duration":"common",
        "integration_frequency":integration_freq_sec,
        "param":{},
        "method":"meta"
    }

    return integration_param
 