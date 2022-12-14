from Clust.clust.meta.metaDataManager import wizMongoDbApi as wiz
from Clust.setting import influx_setting_KETI as ins
from Clust.clust.ingestion.influx import influx_Client_v2 as iC   

def getDataBaseInstance(self): 
  
        mongo_server_address    = ins.wiz_url #몽고 디비 아이피 주소
        mongo_instance          = wiz.WizApiMongoMeta(mongo_server_address)
        influx_instance         = iC.influxClient(ins.CLUSTDataServer2) 
        
        return {'influx_db' : influx_instance, 'mongo_db' : mongo_instance, 'mongo_ip_addr' : mongo_server_address}