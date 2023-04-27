import sys
sys.path.append("../")
from Clust.app import  data_preprocessing
def save_integrated_data_meta_by_clean_level(ms_name, bucket_name, db_client, ingestion_type, ingestion_param, 
                                             clean_level, processing_type, integration_param,
                                             data_name, mongo_client):
    # Processing Param by clean_level
    process_param = data_preprocessing.get_process_param_by_level(clean_level)
    
    # meta info
    meta_info={"data_name": ms_name,"ingestion_param": ingestion_param, 
               "integration_param":integration_param, "clean_level":clean_level, "process_param":process_param}
    collection_name = "forecasting_"+data_name
    ##########################################          
    # 1. Data Manipulation: dataset ingestion---> data preprocessing ---> data integration 
    data = ingestion_processing_integration(db_client, ingestion_type, ingestion_param, processing_type, process_param, integration_param)
    # 2. Save Data##########################
    db_client.write_db(bucket_name, ms_name, data)
    # 3. Save Meta########################## 
    mongo_client.insert_document(bucket_name, collection_name, meta_info)
    
def ingestion_processing_integration(db_client, ingestion_type, ingestion_param, processing_type, process_param, integration_param):
    
    from Clust.clust.data import data_interface
    multiple_dataset = data_interface.get_data_result(ingestion_type, db_client, ingestion_param)

    # 2. Data Preprocessing
    if process_param is None:
        pass
    else:
        from Clust.clust.preprocessing import processing_interface
        multiple_dataset = processing_interface.get_data_result(processing_type, multiple_dataset, process_param)

    # 3. Data Integration
    from Clust.clust.integration.integrationInterface import IntegrationInterface
    
    data = IntegrationInterface().multipleDatasetsIntegration(integration_param, multiple_dataset)
    
        
    return data