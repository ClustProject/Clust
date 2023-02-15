from io import BytesIO
import os
import base64
import matplotlib.pyplot as pyplt
from Clust.clust.tool.file_module import file_common

def plt_to_image(plt):
    """
    Convert plt into real image
    Args:
        plt (matplotlib.pyplot):
    Return:
        image_base64 (image_base64)
    """
    # send images
    buf = BytesIO()
    plt.savefig(buf, format='png')
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
    return image_base64


def savePlotImg(plot_name, plot_obj, img_directory) :     
    """
    - Save plot image to image directory
    - 원하는 디렉토리에 플롯 이미지를 저장한다.

    Args :
        plot_name(str)
        plot_obj(pyplot object)
        img_directory(str)
    
    Return :
        None
         
    """    
	
    file_common.check_path(img_directory,(plot_name+'.png'))
    plot_obj.savefig(os.path.join(img_directory,(plot_name+'.png')))  
    plot_obj.clf()


def get_dataset(client_parameter,  db_client) :

    """           
    # Description
    - 파라미터를 받아, EDA 그래프 함수들이 공통적으로 사용하는 DataFrame을 리턴하는 함수

    # Input Data
    * param
        - client에서 받은 파라미터
    ```
        {'db_array': [{'air_indoor_modelSchool': 'ICW0W2000022'}, {'air_outdoor_kweather': 'OC3CL200012'}, {'air_outdoor_keti_clean': 'seoul'}]}
    ```

    # Returns
    * DataFrame         
    ```
        out_SO2   out_h2s   out_pm10
        datetime                                               
        2021-09-05 00:00:00+00:00    0.002  0.000000   1.433333
    ```

    """
    bucket_array        = client_parameter['bucket_array']
    start_time          = client_parameter['start_time']
    end_time            = client_parameter['end_time']
    feature_compared    = client_parameter['feature_compared']            

    ## Static Variables (나중에 입력 받을 수도)        
    integration_freq_sec = int(client_parameter['integration_freq_min']) * 60 # 1시간 
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
    process_param       = {'refine_param':refine_param, 'outlier_param':outlier_param, 'imputation_param':imputation_param}
    integration_param   = {
        "integration_duration":"common",
        "integration_frequency":integration_freq_sec,
        "param":{},
        "method":"meta"
    }

    from Clust.clust.integration.utils import param
    intDataInfo = param.makeIntDataInfoSet(bucket_array, start_time, end_time)       

    from Clust.clust.ingestion.influx import ms_data
    multiple_dataset  = ms_data.get_only_numericData_in_ms(db_client, intDataInfo)                     

    # data Integration
    from Clust.clust.integration.integrationInterface import IntegrationInterface
    dataIntegrated = IntegrationInterface().multipleDatasetsIntegration(process_param, integration_param, multiple_dataset)

    dataSet_df = dataIntegrated[feature_compared]           

    return dataSet_df