import sys, os
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")

from Clust.setting import influx_setting_KETI as ins
from Clust.clust.ingestion.influx import influx_Client_v2 as influx_Client
db_client = influx_Client.influxClient(ins.CLUSTDataServer2)

DataMetaPath = "./integratedData.json"
scalerRootDir ='./scaler/'
trainModelMetaFilePath ="./model.json"
IntDataFolderName = "data_integrated_result"

current = os.getcwd()
dataFolderPath = os.path.join(current, IntDataFolderName)