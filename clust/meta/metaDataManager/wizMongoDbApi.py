# backend run.py 실행 후 http://localhost:5000/rest/1.0 실행
import requests
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

class WizApiMongoMeta():
    """
        MongoDB에서 메타 데이터를 읽기, 저장 등과 같은 메타 관리를 할 수 있는 Rest API
        
        - MongoDB에 저장된 데이터베이스 리스트 출력 가능
        - MongoDB에 저장된 특정 데이터베이스의 Collection 리스트 출력 가능
        - MongoDB에 저장된 특정 데이터베이스, Collection의 Table 리스트 출력 가능
        - MongoDB에 메타 데이터 저장 가능
            - 단 하나의 메타 데이터 저장 가능 (데이터베이스, Collectino, Table 지정)
            - 특정 데이터베이스, Collection의 여러 메타 데이터 저장 가능
        - MongoDB에 메타 데이터 읽기 가능
            - 단 하나의 메타 데이터 읽기 가능 (데이터베이스, Collectino, Table 지정)
            - 특정 데이터베이스, Collection의 여러 메타 데이터 읽기 가능
    """
    
    def __init__(self, mongodb_instance_url):
        """
        :param mongodb_instance_url: MongoDB의 Instance 주소
        :type mongodb_instance_url: string
        """
        self.mongodb_instance_url = mongodb_instance_url
        
    # get - database list
    def get_database_list(self):
        url = self.mongodb_instance_url+"/rest/1.0/mongodb/{}".format("databases")
        header = {'accept': 'application/json'}
        response = requests.get(url, headers=header)
        print(response.status_code)
        text = response.text

        return json.loads(text)
    
    def get_collection_list(self, domain):
        url = self.mongodb_instance_url+"/rest/1.0/mongodb/collections/{}".format(domain)
        header = {'accept': 'application/json'}
        response = requests.get(url, headers=header)
        print(response.status_code)
        text = response.text

        return json.loads(text)

    def get_tableName_list(self, domain, subdomain):
        url = self.mongodb_instance_url + "/rest/1.0/mongodb/tableNames/{}/{}".format(domain, subdomain)
        header = {'accept': 'application/json'}
        response = requests.get(url, headers=header)
        #print(response.status_code)
        text = response.text

        return json.loads(text)

    # get - database/collection/document?table_name - 지정 table name 출력
    def read_mongodb_document_by_get(self, domain, subdomain, tableName=None):
        # TODO 아래 주석 수정할 것
        """
        :param domain: database
        :type domain: string

        :param subdomain: database
        :type subdomain: string

        :param tableName: database
        :type tableName: string or None

        - type(tableName) == string : 특정 table만 읽어옴
        - tableName == None : domain/subdomain(Collection) 아래 모든 table을 읽어옴

        :return: result
        :rtype: dictionary (single) or array (multiple)

        """

        if tableName: #one document
            url = self.mongodb_instance_url+"/rest/1.0/mongodb/document/{}/{}?table_name={}".format(domain, subdomain, tableName)
        else: #all documents under domain/subdomain/
            url = self.mongodb_instance_url+"/rest/1.0/mongodb/documents/{}/{}".format(domain, subdomain)

        response = requests.get(url)
        print(response.status_code)
        text = response.text
        result = json.loads(text)

        return result

    # post - database/collection/document insert, save
    def save_mongodb_document_by_post(self, mode, data, domain, subdomain):
        """
        mongodb의 document를 post로 저장함

        :param mode: data를 mongodb에 처리 하기 위한 방법 [update|insert|save]
        :type mode: string

        :param data: mongodb에 처리할 data
        :type data: array[dictionaries]

        :param domain: domain, mongodb의 database 이름
        :type domain: string

        :param subdomain: subdomain, mongodb의 collection 이름
        :type subdomain: string

        """
        headers = {'Content-Type': 'application/json'}
        
        url = self.mongodb_instance_url+"/rest/1.0/mongodb/documents/{}/{}?mode={}".format(domain, subdomain, mode)
        response = requests.post(url, data=json.dumps(data), headers=headers)
        print("Success:", response.status_code)

if __name__ == "__main__":
    from pprint import pprint
    from Clust.setting import influx_setting_KETI as ins
    import json
    
    mongo_instance = ins.wiz_url
    test = WizApiMongoMeta(mongo_instance)
    meta = test.read_mongodb_document_by_get("air", "indoor_유치원", "ICW0W2000132")
    print(meta)
    '''
    get - database/collection/document - 첫번째 document 만 출력
    get - database/collection/documents 는 head 5개만 출력 -> 갯수 변경하고 싶을 시 limit 수정하면 가능
    get - database/collection/document?table_name = 지정해주기 -> 원하는 table name의 document 출력

    post - 새롭게 document 를 생성할땐 insert
    post - 기존에 있는 document에 수정을 위해 덮어씌우기는 save
    post - document에는 필수적으로 "table_name" 을 기입해야한다

    '''