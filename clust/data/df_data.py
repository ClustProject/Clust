import sys, pandas
sys.path.append("../")
sys.path.append("../../")

class DfData():
    def __init__(self, db_client):
        """
        This class makes dataframe style data based on ingestion type, and param

        Args:
            db_client (_instance_): Instance of InfluxClient class. Instance to get data from influx DB.

        """
        self.db_client = db_client
        
    def get_result(self, ingestion_type, ingestion_param):
        """       
        Get dataframe result according to intestion_type, and ingestion_param

        ArgS:
            ingestion_type (_String_) : ingestion_type (method)
            ingestion_param (_Dictionary_) : ingestion parameter depending on ingestion_type

        Returns:
            _DataFrame_ : result

        >>> ingestion_type_list = ['ms_by_days', 'ms_by_time', 'ms_by_num']
        >>> ingestion_type = 'ms_by_num'
        >>> ingestion_param = {
        ...                     'bucket_name' : bucket_name,
        ...                     'ms_name' : ms_name,
        ...                     'num' : 1000,
        ...                     'position' : 'end',
        ...                     'feature_list' : feature_list }  

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
        elif ingestion_type == 'ms_all':
            result = self.ms_by_all(self.ingestion_param) 
        
        if 'feature_list' in ingestion_param.keys():
            if len(result)>0:
                new_feature_list = list(self.ingestion_param['feature_list'].intersection(list(result.columns)))
                result = result[new_feature_list]
            
        return result
    
    def ms_by_all(self, ingestion_parm):
        """
        Get all data ingestion

        Args:
            ingestion_param (_Dictionary_) : get data by all parameter 

        Returns:
            _pd.DaraFrame_ :data(result data)

        >>> ingestion_param = {
        ...                     'bucket_name' : 'air_indoor_modelSchool',
        ...                     'ms_name' : 'ICW0W2000014',
        ...                     'feature_list' : ['CO2'] }
        """
        data = self.db_client.get_data(ingestion_parm["bucket_name"], ingestion_parm['ms_name'])

        return data
        
    def ms_by_days(self, ingestion_param):
        """
        Get data by days 

        Args:
            ingestion_param (_Dictionary_) : data by days parameter

        Returns:
            _DataFrame_ : data(result data)

        Example     

        >>> ingestion_param = {
        ...                     'days' : 1,                                     
        ...                     'end_time': '2021-09-11 00:00:00', 
        ...                     'bucket_name' : 'air_indoor_modelSchool',
        ...                     'ms_name' : 'ICW0W2000014',
        ...                     'feature_list' : ['CO2', 'Noise', 'PM10', 'PM25', 'Temp', 'VoCs', 'humid',
        ...                                         'out_h2s','out_humi', 'out_noise', 'out_temp',
        ...                                         'out_ultraviolet_rays', 'out_PM10','out_PM25'] }

        """
        data = self.db_client.get_data_by_days(ingestion_param['end_time'], ingestion_param['days'], ingestion_param["bucket_name"], ingestion_param['ms_name']) 

        return data
    
    def ms_by_time(self, ingestion_param):
        """
        Get data by time duration

        Args:
            ingestion_param (_Dictionary_) : data by time duration parameter 

        Returns:
            _DaraFrame_ : data(result data)


        >>> ingestion_param = {
        ...                     'bucket_name' : 'air_indoor_modelSchool', 
        ...                     'ms_name' : 'ICW0W2000014', 
        ...                     'start_time': '2021-09-05 00:00:00',
        ...                     'end_time': '2021-09-11 00:00:00',
        ...                     'feature_list' : [ 'CO2', 'Noise','PM10','PM25', 'Temp', 'VoCs', 'humid' ] }

        """
        data = self.db_client.get_data_by_time(ingestion_param['start_time'], ingestion_param['end_time'], ingestion_param["bucket_name"], ingestion_param['ms_name'])
        
        return data
    
    def ms_by_num(self, ingestion_param):
        """
        Get data by num 

        Args:
            ingestion_param (_Dictionary_) : data by num parameter

        Returns:
            _DataFrame_ : data(result data)

        >>> ingestion_param = {
        ...                     'bucket_name' : 'air_indoor_modelSchool',
        ...                     'ms_name' : 'ICW0W2000014',
        ...                     'num' : 1000, #ms_by_num
        ...                     'position' : 'end',
        ...                     'feature_list' : [ 'CO2', 'Noise','PM10','PM25', 'Temp', 'VoCs', 'humid' ] }

        """
        if ingestion_param['position']=='end':
            data = self.db_client.get_data_end_by_num(ingestion_param['num'], ingestion_param["bucket_name"], ingestion_param['ms_name']) 
        else:
            data = self.db_client.get_data_front_by_num(ingestion_param['num'], ingestion_param["bucket_name"], ingestion_param['ms_name']) 
            
        return data             
