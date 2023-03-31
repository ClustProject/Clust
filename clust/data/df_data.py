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
        
    def get_result(self, ingestion_type, ingestion_param) -> pandas.DataFrame :
        """       
        # Description
         get dataframe result according to intestion_type, and ingestion_param

        # Args
         * ingestion_type (_str_): ingestion_type (method)
         ```
            >>> ingestion_type_list = ['ms_by_days', 'ms_by_time', 'ms_by_num']  
         ```
         * ingestion_param (_Dict_): ingestion parameter depending on ingestion_type
         * process_param (_Dict_, _optional_): data preprocessing paramter. Defaults to None.
        # Returns
         * result (_dataframe_)

        """

        # define param
        self.ingestion_param    = ingestion_param
        
        # get result
        if ingestion_type == "ms_by_num":
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
                                , 'num' : 1000 #ms_by_num
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
