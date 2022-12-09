import pymongo
import os, sys
import json
from bson.json_util import dumps, loads
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
from Clust.setting import influx_setting_KETI as ins

class mongoClient():

    def __init__(self,mongo_setting):
        self.mongo_setting = mongo_setting
        self.DBClient = pymongo.MongoClient(host=self.mongo_setting["host"], port=self.mongo_setting["port"], username=self.mongo_setting["username"], password=self.mongo_setting["password"], authSource="admin")


## ------------------------------ Get Function ------------------------------
    def get_DBList(self):
        """
        Get All Mongo Database List

        Returns:
            List: db_list
        """
        db_list = self.DBClient.list_database_names()

        db_list = [bk for bk in db_list if bk not in ['admin', 'config', 'local']]
        # db_list.remove('admin')
        # db_list.remove('config')
        # db_list.remove('local')

        return db_list


    def get_CollectionList(self, db_name):
        """
        Get All Collection list of specific Database

        Args:
            db_name (string): databse

        Returns:
            List: collection list
        """
        database = self.DBClient.get_database(db_name)
        collection_list = database.list_collection_names()

        return collection_list


    def get_all_document(self, db_name, collection_name):
        """
        
        """
        database = self.DBClient.get_database(db_name)
        cursor = database[collection_name].find()
        document_list = list(cursor)

        for cursor_info in document_list:
            del(cursor_info['_id'])

        document_list = loads(dumps(document_list))

        return document_list

        
    def get_document_by_json(self, db_name, collection_name, search):
        """
        
        """
        database = self.DBClient.get_database(db_name)
        cursor = database[collection_name].find(search)
        document_list = list(cursor)

        for cursor_info in document_list:
            del(cursor_info['_id'])

        document_list = loads(dumps(document_list))

        return document_list     


    def get_document_by_table(self, db_name, collection_name, table_name):
        """
        
        """
        database = self.DBClient.get_database(db_name)
        table_Info = {'table_name': table_name}
        cursor = database[collection_name].find(table_Info)
        document_list = list(cursor)

        for cursor_info in document_list:
            del(cursor_info['_id'])
        
        document_list = loads(dumps(document_list))

        return document_list







## ------------------------------ Create & Insert Function ------------------------------
    def insert_document(self, db_name, collection_name, document):
        database = self.DBClient[db_name]
        collection = database[collection_name]

        collection.insert_one(document)
        print("Success")
        







## ------------------------------ Delete Function ------------------------------
    def delete_database(self, db_name):
        """
        Delete Database

        Args:
            db_name (string): databse
        """
        self.DBClient.drop_database(db_name)
        print("Mongo Database Delete Success")


    def delete_collection(self, db_name, collection_name):
        """
        Delete Collection of specific Database

        Args:
            db_name (string): databse
            collection_name (string): collection

        """
        database = self.DBClient.get_database(db_name)
        database[collection_name].drop()
        print("Mongo Collection Delete Success")


    def delete_document(self, db_name, collection_name, document=None):
        """
        Delete Document.

        - document = None: delete all document
        - document = {...}: delete some({...}) document
        """
        database = self.DBClient.get_database(db_name)
        collection = database[collection_name]

        if document == None:
            collection.delete_many({})
            print("Delete All Document Success")
        else:
            collection.delete_many(document)
            print("Delte Some Document Success")    


    def delete_one_document(self, db_name, collection_name, document):
        """
        Delete One Document

        """
        database = self.DBClient.get_database(db_name)
        collection = database[collection_name]

        collection.delete_one(document)
        print("Delte One Document Success")








        # if limit == None:
        #     cursor = db[collec_name].find()
        # else:
        #     cursor = db[collec_name].find().limit(limit)

if __name__ == "__main__":

    test = mongoClient(ins.CLUSTMetaInfo2)

    print("================================ get_DBList ==========================================\n")
    # aa = test.get_DBList()
    # print(aa)

    db_name = 'test'
    collection_name = 'abcdef'
    aaa = test.get_all_document(db_name, collection_name)
    print(type(aaa))
    print(aaa)