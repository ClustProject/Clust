import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from Clust.clust.preprocessing import dataPreprocessing
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
            "removeDuplication":{"flag":True},
            "staticFrequency":{"flag":True, "frequency":None}
        }
        
        outlier_param  = {
            "certainErrorToNaN":{"flag":True},
            "unCertainErrorToNaN":{
                "flag":False,
                "param":{"neighbor":0.5}
            },
            "data_type":"air"
        }

        imputation_param = {
            "serialImputation":{
                "flag":True,
                "imputation_method":[{"min":0,"max":20,"method":"linear" , "parameter":{}}],
                "totalNonNanRatio":70
            }
        }
        self.process_param = {'refine_param':refine_param, 'outlier_param':outlier_param, 'imputation_param':imputation_param}
        
    def set_process_param(self, new_process_param):
        """
        Process Param을 새롭게 정의
        
        :param new_process_param: 새롭게 정의할 processParam
        :type new_process_param: dictionary
        """
        self.process_param = new_process_param
    
    def get_bucket_meta(self, domain, sub_domain, mongo_instance):
        """
        bucket meta를 읽기
        
        :param domain: domain
        :type domain: string

        :param domain: sub_domain
        :type domain: string

        :param domain: mongo_instance
        :type domain: string

        :returns: bucket_meta 정보
        :rtype: dictionary

        """
        db_name ="bucket"
        collection_name = "meta_info"
        mongodb_c = wiz.WizApiMongoMeta(mongo_instance)
        bucket_meta = mongodb_c.read_mongodb_document_by_get(db_name, collection_name, domain+'_'+sub_domain)

        return bucket_meta
    
    def get_ms_meta(self, domain, sub_domain, mongo_instance, table_name):
        """
        MS meta를 읽기
        
        :param domain: domain
        :type domain: string

        :param sub_domain: sub_domain
        :type sub_domain: string

        :param mongo_instance: mongo_instance
        :type mongo_instance: string
        
        :param table_name: table_name
        :type table_name: string

        :returns: ms_meta 정보
        :rtype: dictionary
        """
        mongodb_c = wiz.WizApiMongoMeta(mongo_instance)
        ms_meta = mongodb_c.read_mongodb_document_by_get(domain, sub_domain, table_name)
        
        return ms_meta

    def get_ms_data_by_days(self, bucket_name, measurement_name, influx_instance):
        """
        Influx에서 1년 기간의 시계열 데이터 인출하고 전처리 하여 전달

        :param bucket_name: domain_subdomain 으로 Influx DB의 데이터베이스 이름
        :type bucket_name: string

        :param measurement_name: Measuremnt 명(데이터 명)
        :type measurement_name: string

        :param influx_instance: influx_instance
        :type influx_instance: instance of influxClient class

        :returns: 결과 데이터
        :rtype: dataframe 

        """
        days = 365
        end_time = influx_instance.get_last_time(bucket_name, measurement_name)
        data_nopreprocessing = influx_instance.get_data_by_days(end_time, days, bucket_name, measurement_name)
        # preprocessing
        partialP = dataPreprocessing.DataProcessing(self.process_param)
        dataframe = partialP.all_preprocessing(data_nopreprocessing)["imputed_data"]

        return dataframe
    
    def get_ms_data(self, bucket_name, measurement_name, influx_instance):
        """
        Influx에서 전체 기간의 시계열 데이터 인출하고 전처리 하여 전달

        :param bucket_name: domain_subdomain 으로 Influx DB의 데이터베이스 이름
        :type bucket_name: string

        :param measurement_name: Measuremnt 명(데이터 명)
        :type measurement_name: string

        :param influx_instance: influx_instance
        :type influx_instance: instance of influxClient class

        :returns: 결과 데이터
        :rtype: dataframe 

        """
        data_nopreprocessing = influx_instance.get_data(bucket_name, measurement_name)
        # preprocessing
        partialP = dataPreprocessing.DataProcessing(self.process_param)
        dataframe = partialP.all_preprocessing(data_nopreprocessing)["imputed_data"]

        return dataframe
    
