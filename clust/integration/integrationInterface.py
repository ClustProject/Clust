from functools import partial
import sys
sys.path.append("../")
sys.path.append("../..")
sys.path.append("../../..")
import datetime
from Clust.clust.integration.meta import partialDataInfo
from Clust.clust.preprocessing import dataPreprocessing
from Clust.clust.ingestion.influx import multipleDataSets
from Clust.clust.integration.ML import RNNAEAlignment
from Clust.clust.integration.meta import data_integration


class IntegrationInterface():
    """
    Data Integration Class
    """
    def __init__(self):
        pass

    def integrationByInfluxInfo(self, db_client, intDataInfo, process_param, integration_param):
        """ 
        Influx에서 데이터를 직접 읽고 Parameter에 의거하여 데이터를 병합

        1. intDataInfo 에 따라 InfluxDB로 부터 데이터를 읽어와 DataSet을 생성함
        2. multipleDatasetsIntegration 함수를 이용하여 결합된 데이터셋 재생성

        Args:
            intDataInfo (json): 병합하고 싶은 데이터의 정보로 DB Name, Measuremen Name, Start Time, End Time를 기입
            process_param (json): Refine Frequency를 하기 위한 Preprocessing Parameter
            integration_param (json): Integration을 위한 method, transformParam이 담긴 Parameter
            
        Example:

        >>> intDataInfo = { "db_info":[ 
        ...     {"db_name":"farm_inner_air", "measurement":"HS1", "start":start_time, "end":end_time},
        ...     {"db_name":"farm_outdoor_weather_clean", "measurement":"gunwi", "start":start_time, "end":end_time},
        ...     {"db_name":"farm_outdoor_air_clean", "measurement":"gunwi", "start":start_time, "end":end_time},
        ... ]} 

        >>> process_param 
        ... refine_param = {
        ...     "removeDuplication":{"flag":True},
        ...     "staticFrequency":{"flag":True, "frequency":None}
        ... }
        ... CertainParam= {'flag': True}
        ... uncertainParam= {'flag': False, "param":{
        ...         "outlierDetectorConfig":[
        ...                 {'algorithm': 'IQR', 'percentile':99 ,'alg_parameter': {'weight':100}}    
        ... ]}}
        ... outlier_param ={
        ...     "certainErrorToNaN":CertainParam, 
        ...     "unCertainErrorToNaN":uncertainParam
        ... }
        ... imputation_param = {
        ...     "flag":False,
        ...     "imputation_method":[{"min":0,"max":3,"method":"linear", "parameter":{}}],
        ...     "totalNonNanRatio":80
        ... }
        ... process_param = {'refine_param':refine_param, 'outlier_param':outlier_param, 'imputation_param':imputation_param}

        >>> integrationFreq_min= 30
        >>> integration_freq_sec = 60 * integrationFreq_min # 분

        >>> MLIntegrationParamExample = {
        ...     "integration_duration":"total" ["total" or "common"],
        ...     "integration_frequency":"",
        ...     "param":{
        ...                     "model": 'RNN_AE',
        ...                     "model_parameter": {
        ...                         "window_size": 10, # 모델의 input sequence 길이, int(default: 10, 범위: 0 이상 & 원래 데이터의 sequence 길이 이하)
        ...                         "emb_dim": 5, # 변환할 데이터의 차원, int(범위: 16~256)
        ...                         "num_epochs": 50, # 학습 epoch 횟수, int(범위: 1 이상, 수렴 여부 확인 후 적합하게 설정)
        ...                         "batch_size": 128, # batch 크기, int(범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
        ...                         "learning_rate": 0.0001, # learning rate, float(default: 0.0001, 범위: 0.1 이하)
        ...                         "device": 'cpu' # 학습 환경, ["cuda", "cpu"] 중 선택
        ...                     }
        ...                 },
        ...     "method":"ML" #["ML", "meta", "simple]
        ... }

        >>> metaIntegrationParamExample = {
        ...     "integration_duration":"total" ["total" or "common"],
        ...     "integration_frequency":integration_freq_sec,
        ...     "param":{},
        ...     "method":"meta"
        ... }
                
        Returns:
            DataFrame: integrated_data  
        
        """
        ## multiple dataset
        multiple_dataset  = multipleDataSets.get_onlyNumericDataSets(db_client, intDataInfo)
        ## get integrated data
        result = self.multipleDatasetsIntegration(process_param, integration_param, multiple_dataset)

        return result
    
    def multipleDatasetsIntegration(self, process_param, integration_param, multiple_dataset):
        """ 
        dataSet과 Parameter에 따라 데이터를 병합하는 함수

        1. 각 데이터셋이서 병합에 필요한 partial_data_info (각 컬럼들에 대한 특성) 추출
        2. 명확하게 정합 주기 값이 입력 없을 경우 최소공배수 주기를 설정함
        3. 각 데이터들에 대한 Preprocessing  
        4. 입력 method에 따라 3가지 방법 중 하나를 선택하여 정합함

        Args:
            process_param (json): Refine Frequency를 하기 위한 Preprocessing Parameter
            integration_param (json): Integration을 위한 method, transformParam이 담긴 Parameter
            
        Returns:
            DataFrame: integrated_data
    
        """

        integration_duration = integration_param["integration_duration"]
        partial_data_info = partialDataInfo.PartialData(multiple_dataset, integration_duration)
        overlap_duration = partial_data_info.column_meta["overlap_duration"]
        integration_freq_sec = integration_param["integration_frequency"]
        integrationMethod = integration_param['method']

        ## set refine frequency parameter
        if not integration_freq_sec:
            process_param["refine_param"]["staticFrequency"]["frequency"] = partial_data_info.partial_frequency_info['GCDs']

        ## Preprocessing
        partialP = dataPreprocessing.DataProcessing(process_param)
        print("processingStart")
        multiple_dataset = partialP.multiDataset_all_preprocessing(multiple_dataset)
        print("processingEnd")
        ## Integration
        imputed_datas = {}
        print("integrationStart")
        for key in multiple_dataset.keys():
            imputed_datas[key]=(multiple_dataset[key])
        if integrationMethod=="meta":
            result = self.getIntegratedDataSetByMeta(imputed_datas, integration_freq_sec, partial_data_info)
        elif integrationMethod=="ML":
            result = self.getIntegratedDataSetByML(imputed_datas, integration_param['param'], overlap_duration)
        elif integrationMethod=="simple":
            result = self.IntegratedDataSetBySimple(imputed_datas, integration_freq_sec, overlap_duration)
        else:
            result = self.IntegratedDataSetBySimple(imputed_datas, integration_freq_sec, overlap_duration)
        print("integrationEnd")
        return result

    def getIntegratedDataSetByML(self, data_set, transform_param, overlap_duration):
        """ 
        ML을 활용한 데이터 병합 함수
        1. 병합한 데이터에 RNN_AE 를 활용해 변환된 데이터를 반환

        Args:
            data_set (json): 병합하고 싶은 데이터들의 셋
            transform_param (json): RNN_AE를 하기 위한 Parameter
            overlap_duration (json): 병합하고 싶은 데이터들의 공통 시간 구간

        Example:

        >>> overlap_duration = {'start_time': Timestamp('2018-01-03 00:00:00'), 
        ...                     'end_time': Timestamp('2018-01-05 00:00:00')}
                
        Returns:
            DataFrame: integrated_data by transform
        """
        
        ## simple integration
        data_int = data_integration.DataIntegration(data_set)
        dintegrated_data = data_int.simple_integration(overlap_duration)
        
        model = transform_param["model"]
        transfomrParam = transform_param['param']
        if model == "RNN_AE":
            integratedDataSet = RNNAEAlignment.RNN_AE(dintegrated_data, transfomrParam)
        else :
            print('Not Available')
            integratedDataSet= None
            
        return integratedDataSet

    def getIntegratedDataSetByMeta(self, data_set, integration_freq_sec, partial_data_info):
        """ 
        Meta(column characteristics)을 활용한 데이터 병합 함수

        Args:
            data_set (json): 병합하고 싶은 데이터들의 셋
            integration_freq_sec (json): 조정하고 싶은 second 단위의 Frequency
            partial_data_info (json): column characteristics의 info
            
        Returns:
            DataFrame: integrated_data   
        """
        ## Integration
        data_it = data_integration.DataIntegration(data_set)
        
        re_frequency = datetime.timedelta(seconds= integration_freq_sec)
        integratedDataSet = data_it.dataIntegrationByMeta(re_frequency, partial_data_info.column_meta)
        
        return integratedDataSet 
    
    def IntegratedDataSetBySimple(self, data_set, integration_freq_sec, overlap_duration):
        """ 
        Simple한 병합

        Args:
            data_set (json): 병합하고 싶은 데이터들의 셋
            integration_freq_sec (json): 조정하고 싶은 second 단위의 Frequency
            overlap_duration (json): 조정하고 싶은 second 단위의 Frequency
            
        Returns:
            DataFrame: integrated_data
        """
        ## simple integration
        re_frequency = datetime.timedelta(seconds= integration_freq_sec)
        data_int = data_integration.DataIntegration(data_set)
        integratedDataSet = data_int.simple_integration(overlap_duration)
        integratedDataSet = integratedDataSet.resample(re_frequency).mean()
        
        return integratedDataSet





