import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from Clust.clust.meta.metaDataManager import collector
from Clust.clust.meta.analysisMeta.meanAnalyzer import holiday, timeStep, working
from Clust.clust.meta.analysisMeta.simpleAnalyzer import countAnalyzer, statisticsAnalyzer

class analysisMSMetaGenerator():
    """
        MS-분석 A Meta를 생성하는 Generator
    """
    def __init__(self, analysis_param, influx_instance, mongo_instance):
        """
        :param analysis_param: analysis를 위한 param
        :type analysis_param: dictionary

        >>> analysisParam = {
            "dbName": "air",
            "collectionName": "indoor_유치원",
            "measurementList" : None, 
            "functionList" : None
        }
        >>>>>> functionList : Analyzed Method Name 으로 list type으로 받음
        >>>>>>>>> functionList example : None or ["StatisticsAnalyzer", "MeanByHoliday", "MeanByWorking", "MeanByTimeStep", "CountByFeatureLabel"]
        
        :param influx_instance: instance to get data from influx DB
        :type influx_instance: instance of influxClient class

        :param mongo_instance: instance url to get meta data from mongo DB
        :type mongo_instance: string
        """

        self.mongodb_db = analysis_param["dbName"]
        self.mongodb_collection =  analysis_param["collectionName"]
        self.mongo_instance = mongo_instance
        
        self.influx_measurement_list = analysis_param["measurementList"]
        self.influxdb_bucket_name = analysis_param["dbName"]+"_"+analysis_param["collectionName"]
        self.influx_instance = influx_instance
        
        self.function_list = analysis_param["functionList"]
        self.function_list = self.checkFunctionList(self.function_list)

    def checkFunctionList(self, function_list):
        """
        분석 방법 리스트를 체크하는 기능 
        - None으로 기입됐을 경우 5가지 모든 분석을 실행한다는 의미로 function_list 에 5가지 분석 방법을 입력
        
        :param function_list: Analyzed Method Name
        :type function_list: list or None

        :returns: function_list : function_list
        :rtype: list
        """
        if function_list is None:
            function_list = ["StatisticsAnalyzer", "MeanByHoliday", "MeanByWorking", "MeanByTimeStep", "CountByFeatureLabel"]
        
        return function_list

    def get_metaset(self):
        """
        - 각 데이터 Measurement에 따른 분석 결과를 기반으로 분석 메타 셋을 생성

        :returns: analysis_meta_set : 각 테이블에 대한 분석 결과에 따른 테이블
        :rtype: array of dictionaries
        """
        self.influx_measurement_list = self._check_ms_list(self.influx_measurement_list)
        collect = collector.ReadData()
        bucket_meta = collect.get_bucket_meta(self.mongodb_db, self.mongodb_collection, self.mongo_instance)

        self.analysis_meta_set = []
        for measurement in self.influx_measurement_list:
            print(f"====== Analyze Meta By {measurement}... ======")
            data = collect.get_ms_data_by_days(self.influxdb_bucket_name, measurement, self.influx_instance)
            analysis_result_set = self.get_result_set(data, bucket_meta, self.function_list)
            analysis_result_set["table_name"] = measurement
            self.analysis_meta_set.append(analysis_result_set)
            print(f"====== SUCCESS {measurement} ======")
        
        return self.analysis_meta_set

    def _check_ms_list(self, ms_list): 
        """
        - ms_list를 체크하고 None일 경우 전체 Bucket 안의 list를 불러옴

        :param ms_list: ms_list
        :type ms_list: (array of string) or None

        :returns: ms_list : 각 테이블에 대한 분석 결과에 따른 테이블
        :rtype: array of string
        """

        if ms_list is None:
            ms_list = self.influx_instance.measurement_list(self.influxdb_bucket_name)

        return ms_list
    
    def get_result_set(self, data, meta, function_list):
        """
        - functionList에 의거하여 분석 결과를 생성함

        :param data: 개별 MS 데이터
        :type data: pd.DataFrame

        :param meta: Bucket Meta
        :type meta: dictionary

        :param function_list: 수행할 분석 Funcion List
        :type function_list: array of string
        
        :returns: "analysisResult" Key를 갖는 분석 결과
        :rtype: dictionary
        """

        result ={}
        for function in function_list:
            print(f"====== Start {function} Analyzed ======")
            if "StatisticsAnalyzer"  == function:
                result[function] = statisticsAnalyzer.Statistics(data).get_result()
            elif "MeanByHoliday" ==function:
                result[function] = holiday.MeanByHoliday(data).get_result()
            elif "MeanByWorking" ==function:
                result[function] = working.MeanByWorking(data).get_result()
            elif "MeanByTimeStep" ==function:
                result[function] = timeStep.MeanByTimeStep(data).get_result()
            elif "CountByFeatureLabel" ==function:
                result[function] = countAnalyzer.CountByFeatureLabel(data, meta).get_result()

        return {"analysisResult":result}
    

