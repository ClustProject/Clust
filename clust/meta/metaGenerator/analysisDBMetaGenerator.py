import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from Clust.clust.meta.metaDataManager import wizMongoDbApi as wiz

class analysisDBMetaGenerator():
    """
        DB-분석 A Meta를 생성하는 Generator
    """
    def __init__(self, dbName, collectionName, labels, mongo_instance):
        """
        :param dbName: Mongo DB 에서 활용되는 dbName으로 특정 데이터베이스의 명칭
        :type dbName: string
        
        :param collectionName: collectionName
        :type collectionName: string
        
        :param labels: 특정 분석 방법의 결과 label 정보
        :type labels: dictionary

        >>> labels = {
            "StatisticsAnalyzer" : ["min", "max", "mean"],
            "MeanByHoliday" : ["holiday", "notHoliday"],
            "MeanByWorking" : ["working", "notWorking"],
            "MeanByTimeStep" : ["dawn", "morning", "afternoon", "evening", "night"]
        }
        
        :param mongo_instance: instance url to get meta data from mongo DB
        :type mongo_instance: string
        """
        
        self.db_name = dbName
        self.collection_name = collectionName
        self.labels = labels
        self.mongo_instance = mongo_instance
    
    def get_bucketAnalysisMeta(self):
        """
        - 특정 bucket의 하위에 존재하는 각 MS-분석 A 정보들의 평균 값으로 bucket analysis meta 생성

        :returns: bucket_meta : 각 테이블에 대한 분석 결과에 따른 테이블
        :rtype: array of dictionary
        """
        print("=== start ===")
        total_ms_analysis_meta, column_list = self.get_allMsAnalysisMeta(self.db_name, self.collection_name, self.mongo_instance)
        bucket_analysis_meta = self.calculate_msMeta(total_ms_analysis_meta, column_list, self.labels)
        
        bucket_name = self.db_name+"_"+self.collection_name
        bucket_meta = [{"table_name":bucket_name, "analysisResult":bucket_analysis_meta}]
        
        print("=== get bucket_analysis_meta by {} ===".format(bucket_name))
        
        return bucket_meta
        
    def get_allMsAnalysisMeta(self, dbName, collectionName, mongo_instance):
        """
        - bucket analysis meta 생성에 필요한 해당 bucket의 하위에 존재하는 모든 MS-분석 A 정보를 모으는 함수
        
        :param dbName: Mongo DB 에서 활용되는 dbName으로 특정 데이터베이스의 명칭
        :type dbName: string
        
        :param collectionName: collectionName
        :type collectionName: string
        
        :param mongo_instance: instance url to get meta data from mongo DB
        :type mongo_instance: string
        
        :returns: total_ms_analysis_meta : 해당 bucket 하위의 모든 MS-분석 A 정보가 모인 Meta
        :rtype: list of dictionary
        
        :returns: column_list : 해당 bucket 하위의 모든 MS의 Column List
        :rtype: list of string
        """
        mongodb_c = wiz.WizApiMongoMeta(mongo_instance)
        ms_list = mongodb_c.get_tableName_list(dbName, collectionName)[dbName][collectionName]
        
        ms_list.remove("db_information") # db_information 이전 완료하면 삭제하기
        
        total_ms_analysis_meta = []
        column_list = []
        for ms in ms_list:
            ms_meta = mongodb_c.read_mongodb_document_by_get(dbName, collectionName, ms)
            total_ms_analysis_meta.append(ms_meta["analysisResult"])
            column_list.extend([field["fieldKey"] for field in ms_meta["fields"]])
        
        column_list = list(set(column_list))
        
        return total_ms_analysis_meta, column_list
        
    def calculate_msMeta(self, total_ms_analysis_meta, column_list, labels):
        """
        - 모은 MS-분석A 모든 Meta를 기반으로 분석 결과에 따른 평균 값 계산을 하여 최종 bucket_analysis_meta 생성
        
        :param total_ms_analysis_meta: 해당 bucket 하위의 모든 MS-분석 A 정보가 모인 Meta
        :type total_ms_analysis_meta: list of dictionary
        
        :param column_list: 해당 bucket 하위의 모든 MS의 Column List
        :type column_list: list of string
        
        :param labels: 특정 분석 방법의 결과 label 정보
        :type labels: dictionary
        
        :returns: bucket_analysis_meta : 최종 bucket analysis meta
        :rtype: dictionary
        """
        bucket_analysis_meta = []
        
        for analyzer_name in list(labels.keys()):
            for column in column_list:
                analysis_result_dict = {}
                analysis_result_dict["analyzerName"] = analyzer_name
                analysis_result_dict["columnName"] = column
                
                analysis_label = labels[analyzer_name]
                mean_result_list = []
                
                for label in analysis_label:
                    result_value_by_all_ms = []
                    for analyzer_result in total_ms_analysis_meta:
                        try:
                            result_value_by_all_ms.append(analyzer_result[analyzer_name][column][label])
                        except KeyError:
                            continue
                    value = list(map(self.none_convert_nan, result_value_by_all_ms)) # none -> nan (계산을 위해)
                    mean_result_list.append(np.nanmean(value))
                mean_result_list = list(map(self.nan_convert_none, mean_result_list)) # nan -> None (UI를 위해)
                    
                analysis_result_dict["label"] = analysis_label
                analysis_result_dict["resultValue"] = mean_result_list
                bucket_analysis_meta.append(analysis_result_dict)
                
        return bucket_analysis_meta
    
    def none_convert_nan(self,value):
        """
        "None" 으로 저장되었던 Meta 값들을 계산에 용이하게 nan 으로 변형해주는 함수
        
        - Meta 값이 "None"인지 판단 후에 "None"일 경우 nan으로 변형

        Args:
            value (string): 변형해주고 싶은 Meta value로 string type이여야 한다.

        Returns:
            - "None"이면 nan으로 변형된 값
            - "None"이 아니였다면 value 그대로 반환
        """
        if value == "None":
            value = np.nan
        return value
    
    def nan_convert_none(self, value):
        """
        nan 으로 저장되었던 Meta 값들을 "None"으로 변형해주는 함수
        
        - Meta 값이 nan인지 판단 후에 nan일 경우 "None"으로 변형

        Args:
            value (string): 변형해주고 싶은 Meta value로 string type이여야 한다.

        Returns:
            - nan이면 "None"으로 변형된 값
            - nan이 아니였다면 value 그대로 반환
        """
        if np.isnan(value):
            value = "None"
        return value
