import sys
sys.path.append("../")
sys.path.append("../../")

from Clust.clust.ingestion.influx import df_set_data

class DfData():
    def __init__(self, db_client):
        """This class makes dataframe style data based on ingestion type, and param

        Args:
            db_client (_type_): _description_
        """
        self.db_client = db_client
        
    def get_result(self, ingestion_type, ingestion_param, process_param = None):
        """get dataframe result according to intestion_type, and ingestion_param

        Args:
            ingestion_type (string): ingestion_type (method)
            ingestion_param (dictionary): ingestion parameter depending on ingestion_type
            process_param (dictionary, optional): data preprocessing paramter. Defaults to None.

        Returns:
            dataframe: result
        """
        # define param
        self.ingestion_param = ingestion_param
        self.process_param = check_process_param(process_param)
        
        # get result
        if ingestion_type == "multi_ms_integration":
            result = self.multi_ms_integration(self.ingestion_param)  
            
        return result
    
    def multi_ms_integration(self, ingestion_param):
        """ get integrated numeric data with multiple MS data integration: ms1+ms2+....+msN => DF

        Args:
            ingestion_param (dict): ingestion_param 

        >>> ingestion_param = 
        {
        'start_time': '2021-09-05 00:00:00', 
        'end_time': '2021-09-11 00:00:00', 
        'integration_freq_min': 60, 
        'feature_list': ['CO2', out_PM10', 'out_PM25'], 
        'ms_list_info': [['air_outdoor_kweather', 'OC3CL200012'], ['air_outdoor_keti_clean', 'seoul'], ['air_indoor_modelSchool', 'ICW0W2000011']]}


        Returns:
            pd.dataFrame: integrated data
            
        """
        
        integration_freq_sec = int(ingestion_param['integration_freq_min']) * 60 
        integration_param = get_integration_param(integration_freq_sec)
        multiple_dataset  = df_set_data.DfSetData(self.db_client).get_result("multi_numeric_ms_list", ingestion_param)
        # data Integration
        from Clust.clust.integration.integrationInterface import IntegrationInterface
        dataIntegrated = IntegrationInterface().multipleDatasetsIntegration(self.process_param, integration_param, multiple_dataset)
        
        if ingestion_param['feature_list']:
            dataIntegrated = dataIntegrated[ingestion_param['feature_list']]
        
        return dataIntegrated
            
            
def check_process_param(process_param):
    """Check and generate general preprocessing parameter (if not exists)

    Args:
        process_param (dictionary or None): input process param

    Returns:
        dictionary: process_param
    
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

def get_integration_param(integration_freq_sec):
    """Generate general integration parameter

    Returns:
        dictionary: process_param
    """
    integration_param   = {
        "integration_duration":"common",
        "integration_frequency":integration_freq_sec,
        "param":{},
        "method":"meta"
    }
    return integration_param
 