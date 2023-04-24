import sys
sys.path.append("../")

def ingestion_processing_integration(db_client, ingestion_param, process_param, integration_param):
    
    from Clust.clust.data import data_interface
    multiple_dataset = data_interface.get_data_result("multiple_ms_by_time", db_client, ingestion_param)

    # 2. Data Preprocessing
    if process_param is None:
        pass
    else:
        from Clust.clust.preprocessing import processing_interface
        multiple_dataset = processing_interface.get_data_result('step_3', multiple_dataset, process_param)

    # 3. Data Integration
    from Clust.clust.integration.integrationInterface import IntegrationInterface
    
    data = IntegrationInterface().multipleDatasetsIntegration(integration_param, multiple_dataset)
    
        
    return data