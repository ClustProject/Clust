import sys
sys.path.append("../")
sys.path.append("../../")

from Clust.clust.ingestion.influx import ms_data

class DfSetData():
    def __init__(self,ingestion_type, param, db_client):
        if ingestion_type == "multiMs_MsinBucket":
            result = ms_data.get_integated_multi_ms_and_one_bucket(param, db_client)
        return result 