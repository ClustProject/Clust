
import sys
sys.path.append("../")
from Clust.clust.data import df_set_data

def integration_from_influx_info(db_client, intDataInfo, process_param, integration_param):
    """Influx에서 데이터를 직접 읽고 Parameter에 의거하여 데이터를 병합

    1. intDataInfo 에 따라 InfluxDB로 부터 데이터를 읽어와 DataSet을 생성함
    2. multipleDatasetsIntegration 함수를 이용하여 결합된 데이터셋 재생성함 생성시 stpe_3 preprocessing을 수행

    Args:
        db_client (dbclinet): 데이터를 뽑기 위한 인스턴스
        intDataInfo (json): 병합하고 싶은 데이터의 정보로 DB Name, Measuremen Name, Start Time, End Time를 기입
        process_param (json): Refine Frequency를 하기 위한 Preprocessing Parameter
        integration_param (json): Integration을 위한 method, transformParam이 담긴 Parameter

    Returns:
        result(pd.DataFrame): integrated_data  
    """

    
    ## multiple dataset
    multiple_dataset  = df_set_data.DfSetData(db_client).get_result("multiple_ms_by_time", intDataInfo)
    ## get integrated data
    ## Preprocessing
    from Clust.clust.preprocessing import processing_interface
    multiple_dataset = processing_interface.get_data_result('step_3', multiple_dataset, process_param)
    
    from Clust.clust.integration import integration_interface
    result = integration_interface.get_data_result('multiple_dataset_integration', multiple_dataset, integration_param)


    return result