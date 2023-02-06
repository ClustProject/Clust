
import os

class plot_modules():
        def get_dataset(param,  db_client) :
            """           
            # Description
            - 그래프 생성에 필요한 데이터 셋을 반환한다.

            # Input Data
            * param
            ```
                {   
                    integration_freq_min: '60', 
                    lag_number: '24', 
                    start_time: '2021-09-05 00:00:00', 
                    end_time: '2021-09-11 00:00:00', 
                    dataInfo: Array(3), …
                }
            ```

            # Returns
            * json         

            """
            dataInfo        = param['dataInfo']
            start_time      = param['start_time']
            end_time        = param['end_time']
            feature_compared = param['feature_compared']            

            ## Static Variables (나중에 입력 받을 수도)        
            integration_freq_sec = int(param['integration_freq_min']) * 60 # 1시간 
            refine_param        = {"removeDuplication":{"flag":True},"staticFrequency":{"flag":True, "frequency":None}}
            
            CertainParam        = {'flag': True}
            uncertainParam      = {'flag': False, "param": {"outlierDetectorConfig":[{'algorithm': 'IQR', 'percentile':99 ,'alg_parameter': {'weight':100}}]}}
            outlier_param       = {"certainErrorToNaN":CertainParam, "unCertainErrorToNaN":uncertainParam}
            imputation_param    = {
                "flag":True,
                "imputation_method":[{"min":0,"max":3,"method":"linear", "parameter":{}}],
                "totalNonNanRatio":80
            }
            process_param       = {'refine_param':refine_param, 'outlier_param':outlier_param, 'imputation_param':imputation_param}
            integration_param   = {
                "integration_duration":"common",
                "integration_frequency":integration_freq_sec,
                "param":{},
                "method":"meta"
            }

            from Clust.clust.integration.utils import param
            intDataInfo = param.makeIntDataInfoSet(dataInfo, start_time, end_time)       

            from Clust.clust.ingestion.influx import ms_data
            multiple_dataset  = ms_data.get_only_numericData_in_ms(db_client, intDataInfo)               
            
            # data Integration
            from Clust.clust.integration.integrationInterface import IntegrationInterface
            dataIntegrated = IntegrationInterface().multipleDatasetsIntegration(process_param, integration_param, multiple_dataset)
            
            dataSet = dataIntegrated[feature_compared]
            

            return dataSet

        def check_path(directory, file_name) :               
        
            if not os.path.exists(directory):
                os.makedirs(directory)
        
            if os.path.exists(file_name):
                os.remove(file_name)