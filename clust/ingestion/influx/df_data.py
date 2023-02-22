import sys
sys.path.append("../")
sys.path.append("../../")

from Clust.clust.ingestion.influx import ms_data


class DfData():
    def __init__(self,ingestion_type, param, db_client):
        if ingestion_type == "multiMS":
            result = ms_data.get_integated_multi_ms(param, db_client)    
        return result