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


    def get_DBList(self):
        """
        Get All Mongo Database List

        Returns:
            List: db_list
        """
        db_list = self.DBClient.list_database_names()
        db_list.remove('admin')
        db_list.remove('config')
        db_list.remove('local')

        return db_list


    def get_Collection(self, db_name):
        """
        Get All Collection list of specific Database

        Args:
            db_name (string): databse

        Returns:
            List: collection list
        """
        db = self.DBClient.get_database(db_name)
        collection_list = db.list_collection_names()

        return collection_list


    def get_all_document(self, db_name, collection_name):
        """
        
        """
        db = self.DBClient.get_database(db_name)
        cursor = db[collection_name].find()
        document_list = list(cursor)

        for cursor_info in document_list:
            del(cursor_info['_id'])

        return document_list

        
    def get_one_document_by_json(self, db_name, collection_name, search):
        """
        
        """
        db = self.DBClient.get_database(db_name)
        cursor = db[collection_name].find(search)
        document_list = list(cursor)

        for cursor_info in document_list:
            del(cursor_info['_id'])

        return document_list     


    def get_document_by_table(self, db_name, collection_name, table_name):
        """
        
        """
        db = self.DBClient.get_database(db_name)
        table_Info = {'table_name': table_name}
        cursor = db[collection_name].find(table_Info)
        document_list = list(cursor)

        for cursor_info in document_list:
            del(cursor_info['_id'])
        
        # document_list = loads(dumps(document_list[0]))

        return document_list







    def delete_database(self, db_name):
        """
        Delete Database

        Args:
            db_name (string): databse
        """
        self.DBClient.drop_database(db_name)
        print("mongo database delete success")


    def delete_collection(self, db_name, collection_name):
        """
        Delete Collection of specific Database

        Args:
            db_name (string): databse
            collection_name (string): collection

        """
        database = self.DBClient.get_database(db_name)
        database[collection_name].drop()
        print("mongo collection delete success")












        # if limit == None:
        #     cursor = db[collec_name].find()
        # else:
        #     cursor = db[collec_name].find().limit(limit)


## -------------------------------------- Mongo Test --------------------------------------
if __name__ == "__main__":

    test = mongoClient(ins.CLUSTMetaInfo)

    print("================================ get_DBList ==========================================\n")
    aa = test.get_DBList()
    print(aa)

    db_name = 'city'
    print("\n================================= get_Collection =========================================\n")
    bb = test.get_Collection(db_name)
    print(bb)

    print("\n******************************************** get_all_document *************************************************************\n")
    collection_name = 'exhibition_entrance_status'
    # cc = test.get_all_document(db_name, collec_name,2)
    # # print(cc)
    # for i in cc:
    #     print(i)
    #     print("\n************************************************\n")

    # search = {'table_name': '57239b9-2-CO'}
    # dd = test.get_one_document_by_json(db_name, collec_name, search)
    # print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(dd)


    table_name = '57239b9-2-CO'
    print("\n-------------------------------------- get_document_by_table -----------------------------------")
    ee = test.get_document_by_table(db_name, collection_name, table_name)
    print(ee)