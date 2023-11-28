import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from Clust.clust.meta.metaDataManager import wizMongoDbApi as wiz
# packcage : InputSourceController
# class : Collector

class ReadData(): 
    def __init__(self):
        """"
        Data를 상황에 맞게 읽는 함수
        
        - InfluxDB에서 필요한 데이터를 읽어서 이를 전처리한 후 출력 가능
            - 기간을 1년으로 데이터를 읽기 및 전처리 가능
            - 전체 기간으로 데이터를 읽기 및 전처리 가능
        - MongoDB에서 필요한 메타 데이터 읽기 가능
            - MS Meta 읽기 가능
            - Bucket Meta 읽기 가능
        """
        refine_param = {
            "remove_duplication":{"flag":True},
            "static_frequency":{"flag":True, "frequency":None}
        }
        
        outlier_param  = {
            "certain_error_to_NaN":{"flag":True},
            "uncertain_error_to_NaN":{
                "flag":False,
                "param":{"neighbor":0.5}
            },
            "data_type":"air"
        }

        imputation_param = {
            "flag":True,
            "imputation_method":[{"min":0,"max":20,"method":"linear" , "parameter":{}}],
            "total_non_NaN_ratio":70
        }
        self.process_param = {'refine_param':refine_param, 'outlier_param':outlier_param, 'imputation_param':imputation_param}
        
    def set_process_param(self, new_process_param):
        """
        Process Param을 새롭게 정의하는 함수
        
        Args:
            new_process_param (_dictionary_) : 새롭게 정의할 processParam

        """
        self.process_param = new_process_param
    
    def get_bucket_meta(self, domain, sub_domain, mongo_instance):
        """
        bucket meta를 가져옴 
        (위즈온텍 API 활용)
        
        Args:
            domain (_str_)    
            sub_domain _str_)    
            mongo_instance (_mongo_instance_)           

        Returns:
            Dictionary : bucket_meta, bucket_meta 정보            

        """
        db_name ="bucket"
        collection_name = "meta_info"
        mongodb_c = wiz.WizApiMongoMeta(mongo_instance)       

        bucket_meta = mongodb_c.read_mongodb_document_by_get(db_name, collection_name, domain+'_'+sub_domain)

        return bucket_meta
    
    def get_ms_meta(self, domain, sub_domain, mongo_instance, table_name):
        """
        MS meta를 가져옴
        
        Args:
            domain (_str_)      
            sub_domain (_str_)    
            mongo_instance (_mongo_instance_)  
            table_name (_str_)            

        Returns:
            Dictionary : ms_meta 정보
    
        """
        mongodb_c = wiz.WizApiMongoMeta(mongo_instance)
        ms_meta = mongodb_c.read_mongodb_document_by_get(domain, sub_domain, table_name)
        
        return ms_meta

    def get_ms_data_by_days(self, bucket_name, measurement_name, influx_instance):
        """
        Influx에서 1년 기간의 시계열 데이터 인출하고 전처리 하여 전달

        Args:
            bucket_name (_str_) : domain_subdomain 으로 Influx DB의 데이터베이스 이름       
            measurement_name (_str_) : Measuremnt 명(데이터 명)
            influx_instance (_influx_instance_) : instance of InfluxClient class            

        Returns:
            pd.dataframe : multiple_dataset(결과 데이터)            

        """
        days = 365
        end_time = influx_instance.get_last_time(bucket_name, measurement_name)
        data_nopreprocessing = influx_instance.get_data_by_days(end_time, days, bucket_name, measurement_name)
        # preprocessing
        from Clust.clust.preprocessing import processing_interface
        multiple_dataset = processing_interface.get_data_result('step_3', data_nopreprocessing, self.process_param)
        
        return multiple_dataset
    
    def get_ms_data(self, bucket_name, measurement_name, influx_instance):
        """
        Influx에서 전체 기간의 시계열 데이터 인출하고 전처리 하여 전달

        Args:
            bucket_name (_str_) : domain_subdomain 으로 Influx DB의 데이터베이스 이름  
            measurement_name (_str_) : Measuremnt 명(데이터 명)
            influx_instance (_influx_instance_) : instance of InfluxClient class         

        Returns:
            pd.dataframe : 결과 데이터            

        """
        data_nopreprocessing = influx_instance.get_data(bucket_name, measurement_name)
        # preprocessing
        from Clust.clust.preprocessing import processing_interface
        multiple_dataset = processing_interface.get_data_result('step_3', multiple_dataset, processing_param)

        return dataframe
    
