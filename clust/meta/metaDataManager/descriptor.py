import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from Clust.clust.meta.metaDataManager import wizMongoDbApi as wiz

def write_data(uploadParam, meta_data, mongo_instance):
    """
    mongoDB에 데이터를 저장함

    Args:
        uploadParam (_dictionary_): 데이터를 쓰기 위해 필요한 파라미터        

    >>> uploadParam = {
        "dbName":"farm",
        "collectionName":"swine_air",
        "mode" : "update"# insert / update / save
    }     

    """

    write_mode = uploadParam["mode"]
    dbName = uploadParam["dbName"]
    collectionName = uploadParam["collectionName"]  

    mongodb_c = wiz.WizApiMongoMeta(mongo_instance)
    mongodb_c.save_mongodb_document_by_post(write_mode, meta_data, dbName, collectionName)
